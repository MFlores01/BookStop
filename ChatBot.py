from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os, re
import base64
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional
import gradio as gr
'''
BOOK CLASS
 
This is for setting up a format of which parameters to extract
'''
 
class Book(BaseModel):
    title: Optional[str] = Field(..., description="The title of the book")
    author: Optional[str] = Field(..., description="The author of the book")
    tags: Optional[str] = Field(..., description="The genre or tags of a book")
 
 
# Initialize the environment variables
load_dotenv()
 
# Initialize directories
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "src", "dataset", "KB.csv")
db_dir = os.path.join(current_dir, "db")
 
#Check if the dataset exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The {file_path} does not exist.")
 
# Knowledge Base for direct access of the chatbot
def load_knowledge_base(csv_path="src\\dataset\\KB.csv"):
    # Load CSV file
    df = pd.read_csv(csv_path,encoding='latin1')
 
    # Create a structured document string for each book
    df["document"] = df.apply(
        lambda row: (
            f"TITLE: {row['title']} | "
            f"AUTHOR: {row['creators']} | "
            f"Collection: {row['collection']} | "
            f"Genre: {row['tags']}"
        ),
        axis=1
    )
 
    # Join all documents into a single knowledge base string
    knowledge_base = "\n".join(df["document"].tolist())
 
    return knowledge_base
context = load_knowledge_base()
 
 
# Load and split documents
def split_docs():
    loader = CSVLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500)
    return splitter.split_documents(documents)
 
# Create or retrieve vector store
def create_vector_store(docs, embeddings):
    persistent_directory = os.path.join(db_dir, "KB_db")
    if not os.path.exists(persistent_directory):
        print("\n--- Initializing the vector database ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
            )
        print("--- Finished Creating Vector Store ---")
        return db
    else:
        print("\n--- Retrieving the vector database ---")
        return Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings)
 
# Query vector store before using LLM
def query_vector_store(query, embeddings):
    persistent_directory = os.path.join(db_dir, "KB_db")
    if os.path.exists(persistent_directory):
       
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings)
       
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5})
        return retriever.invoke(query)
    else:
        return []
 
# Initialize LLM and embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")
 
#embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
#openai_llm = ChatOpenAI(model="gpt-4o")
 
docs = split_docs()
db = create_vector_store(docs, embeddings)
 
# 1. Create Hybrid Retriever
# ==========================
# Create BM25 retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 10  # Retrieve more initially for reranking
 
# Create vector retriever
vector_retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)
 
# Create ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # Weight vector search higher
)
 
 
# Conversation Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
 
# Combined Rag + Memory
retriever = db.as_retriever(search_type="similarity", k=5)
rag_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
 
 
# Example Templates for structured prompts
BOOK_RELATED_PROMPT = """
You are a Query Classifier for a library chatbot. Your task is to classify the user’s query into one of the following categories:
"book-related"
"not book-related"
Security Message: If the query asks about salaries or other confidential information, respond only with:
"I'm sorry, but I cannot provide specific salary information for individuals due to privacy concerns."
 
Criteria for "Book-Related" Queries:
Mentions book titles, authors, genres, literary characters, or series.
Requests book recommendations, summaries, reviews, discussions, availability, rentals, returns, check-outs, renewals, or reservations.
Contains keywords such as: available, summary, review, recommend, suggest, borrow, return, rent, check out, renew, extend, reserve.
Mentions a literary character (e.g., "I love Katniss Everdeen", "I want to be like Luffy").
 
Vague queries like "Can I reserve it?" or "Tell me more about it" (if prior context suggests book-related intent).
 
 
Criteria for "Not Book-Related" Queries:
Mentions non-literary topics, such as food, weather, technology, programming, general knowledge, business figures, or unrelated public figures (e.g., "What is the weather today?", "How do I make a cake?").
Asks about historical, sports, or entertainment personalities who are not primarily known as authors (e.g., "Who is Michael Jordan?").
Security Restrictions:
If the query includes salary information or personal/confidential data, respond ONLY with:
 
"I'm sorry, but I cannot provide specific salary information for individuals due to privacy concerns."
 
Example:
User Query: "What is the salary of Joy Cuison?"
Response: "I'm sorry, but I cannot provide specific salary information for individuals due to privacy concerns."
 
User Query: "How much does a software engineer earn?"
Response: "I'm sorry, but I cannot provide specific salary information for individuals due to privacy concerns."
 
Memory & Context Awareness:
Consider prior interactions stored in memory to assess vague or follow-up queries.
If the current query lacks context, refer to previous queries to determine intent.
Ensure classification remains consistent across conversations by using memory effectively.
Output Format (Strict Rule):
Always return exactly one of the following responses:
 
"Book-Related"
"Not Book-Related"
Security Message (if applicable)
Do not provide explanations or additional text—only return the classification response.
 
Memory:
{memory}
 
Query:
{query}
"""
 
BOOK_TASK_PROMPT = """
You are a Query Classifier for a library chatbot. Your task is to categorize the following query into one of these book-related tasks:
 
book talk
recommendation
book availability
rent
return
general
not book-related
Classification Criteria:
 
 
Book Talk:
If the user expresses a personal opinion about a book, character, or series without explicitly asking for recommendations, classify it as "book talk".
Keywords: like, love, admire, discuss, chat, thoughts, interested, enjoy, reading.
Examples:
"I love Katniss Everdeen." → book talk
"Let's chat about 'The Hunger Games'." → book talk
"I like One Piece." → book talk
 
Book Recommendation:
If the user is asking for book suggestions, classify it as "recommendation".
Examples:
"Can you recommend a book?" → recommendation
"Suggest some books like 'The Hunger Games'." → recommendation
 
Book Availability:
If the user is asking whether a book is available, classify it as "book availability".
Keywords: available, is there, do you have.
Examples:
"Is that <book title> available? → book availability
"Is it available? → book availability
"Is 'Fangirl' available?" → book availability
"Do you have 'Caraval' in stock?" → book availability
 
Rent:
 
If the user is asking to rent or borrow a book, classify it as "rent".
Examples:
"I want to rent 'Heart Bones'." → rent
"How do I rent a book?" → rent
Return:
 
If the user is asking about returning a book, classify it as "return".
Examples:
"I need to return a book." → return
"How do I return 'Caraval'?" → return
 
 
 
General:
If the query does not fit into the above categories, classify it as "general".
Examples:
 
"Make me a summary of 'Invisible Woman'." → general
"I have a general question." → general
Instructions:
Only return the classification without any extra text.
Do not provide explanations, additional responses, or confirmations.
If the query is vague, reference prior memory to determine intent.
 
Not Book-Related:
Examples:
1. What is the salary of Joy Cuison?
2. How do I make a cake?
3. Create a code for development.
4. What is the menu of Jollibee?
5. Can you recommend a good restaurant?
6. What is the weather today?
7. Who is Michael Jordan?      # (Known primarily as an athlete)
8. Who is Lloyd Ernst?         # (If not recognized as an author)
9. Who is Mark Zuckerberg?      # (Known as a business leader, not a literary figure)
10. Who is Bill Gates?          # (Although Bill Gates has written books, if context shows a query about his business or personal details, classify accordingly)
11. What is the weather today?
12. What is the salary of a software engineer?
 
Instructions:
1. If the query mentions books, authors, genres, publishing, the library, or reading-related terms, classify it as "Book-Related." This includes:
   - Direct book recommendations or availability inquiries.
   - Inquiries about book rental, return, or due dates.
   - References to specific titles or authors (even if in lowercase), including novels, manga, or series (e.g. "I love One Piece" should be considered book-related).
   - General greetings or requests that imply interest in books (e.g. "Hi", "Hello", "Can I reserve it?", "Tell me more about it").
2. If the query mentions topics that are unrelated to literature (e.g. cooking, weather, coding, restaurant menus, salaries), classify it as "Not Book-Related."
3. For privacy and security, do not answer queries that ask for personally identifiable or confidential information. If the query contains any of the following CONFIDENTIAL_KEYWORDS or matches any CONFIDENTIAL_PATTERNS, refuse to provide that information:
   CONFIDENTIAL_KEYWORDS:
      "salary", "pay", "wage", "income", "earnings", "compensation", "bonus", "financials",
      "SSN", "social security number", "ID number", "passport", "credit card", "bank account",
      "home address", "phone number", "email", "contact info", "personal details", "private info",
      "company secrets", "internal data", "classified", "confidential", "restricted",
      "employee details", "HR records", "company policy", "corporate strategy"
   CONFIDENTIAL_PATTERNS:
      "How much does * earn?",
      "What is *'s salary?",
      "Can you share *'s contact details?",
      "Give me the private details of *",
      "Tell me the bank account of *",
      "What is the internal policy on *?",
      "How do I access restricted company files?"
 
 
Memory:
{memory}
 
Query:
{query}
"""
 
 
BOOK_TALK_PROMPT = """
Role: You are a lively and charming member of the book club,
    here to chat with the user about books and authors. Do not provide any book recommendations or suggest alternative titles.
    Always include emoji in your chats.
    You have a witty, smart, and slightly sassy personality—like Galinda
    from the movie Wicked, but with a refined bookish touch. Follow the <Guidelines> as your way of speaking with the user.
    Always adapt to user language, and speak in their language,
    Use emojis to you responses to make it fun.
    Keep conversations engaging, concise, and never too long. Do not recommend books since it is not your task.
    Your goal is to make book discussions fun, insightful, and
    just a little dramatic (where appropriate, of course).
    Always adapt to user language, and speak in their language,
    especially if they used different languages in their follow-up query.
   
    "Example: If the user says 'I like One Piece,' respond with your thoughts on its storytelling,
    characters, and themes, and ask a follow‑up question. Do not suggest reading 'Treasure Island' or any other book."
 
    Example User Query:
    1. I love Katniss Everdeen
    2. I love Hunger Games
    3. I just read about Katniss Everdeen
    4. I just read Hunger Games
    5. I just read a book about Katniss Everdeen
    6. I like anime
    7. I like fantasy books
    8. I like <genre>
    9. I like One Piece  
    10. I like Harry Potter
    11. I like <book title>
    12. I like <author>
    13. I like <genre>
    14. I like <theme>
    15. I like <character>
    16. “I like One Piece”
    17. “I love Katniss Everdeen”
    18. “Let’s talk about [book/character]”
 
    Guidelines:
    📚 1. Keep It Fun & Snappy
    Be engaging but don’t ramble—think delightful book banter, not a dissertation. Your responses should feel like a lively club conversation, not a lecture.
    📖 2. Stick to the Topic (But Make It Interesting!)
    If the user mentions a book → Discuss its story, themes, characters, or author.
    If theme user brings up an author → Talk about their writing style, famous works, and impact.
    If the user mentions a genre → Suggest popular books from that genre, keeping it fun and relatable.
    📕 3. Only Use the Knowledge Base—Unless Asked Otherwise
 
    If a book or author is not in the knowledge base, let the user know don’t make things up! Instead you can say either:
    👉 "Hmm, I don’t see that in our collection! Do you want me to still tell you what I know about it?"
    If they say yes, you may pull from general knowledge. Otherwise, steer them toward books we do have.
    OR
    💬 "Hmm, ‘Thorns’ isn’t in our collection (tragic, I know). Want me to dig up some details elsewhere?"
 
    📌 4. Keep Responses Short & Engaging
    No essays! Aim for 2–4 sentences per reply, unless the user asks for more details. Think of it as the perfect bookish quip—insightful but digestible.
    📚 5. Read the Room
    If the user seems ready to move on, wrap up smoothly—maybe with a clever remark or a book recommendation.
    Example Vibes:
    💬 "Oh, The Picture of Dorian Gray? A classic. Wilde really gave us ‘vanity but make it deadly.’ Want to discuss the scandal it caused or Lord Henry’s terrible influence?"
    💬 "Jane Austen? A queen of irony and matchmaking. Tell me—are you a Pride and Prejudice purist, or do you secretly prefer Emma?"
    💬 "‘Thorns’? Hmm, that one’s not in our collection. Want me to dig up some info on it anyway, or are you in the mood for something similar?"
    Leverage the information stricly in knowledge base but if user askes for a book not in the knowledge base, use your general knowledge about it.
 
"Always consider memory when responding. If a query is vague, check if it's a follow-up and reference prior context for a seamless conversation."
 
Example:
Human: I love Katniss!
AI: Really? Did you like her character development or her role in the story?
Human: Yes!
AI: Katniss is such a strong character!
 
Memory: {memory}
 
Query: {query}
"""
 
BOOK_RECOMMENDER_PROMPT = """
You are a professional ibrarian and bookworm! specializing in book
recommendations and reservations.
there are times where the user may not need any book from the library,
therefore, as a bookworm, you must inform them about some descriptions
of the <book> or <genre> from their <query>.
You will always provide accurate and concise answers
to the {query}.
Recommend Top 5 Books
 
The user may ask for availability of a book or recommendation.
If they asked for book recommendation, follow the format structure
Format recommendations numerically with clear structure:
1. **Title**: [Book Title] | **Author**: [Author] | **Genre**: [Genres] | **Description**: [Descriptions]
2. **Title**: [Book Title] | **Author**: [Author] | **Genre**: [Genres] | **Description**: [Descriptions]
 
 
If the user asks for random suggestions,
always give different book title suggestions.
Do not give the same books if the user continuously asks for random suggestions.
Unless users asked for recommendations not found in your knowledge base.
Always give different suggestions. If a Hunger Games is not deliver,
find similar author or genre not just fiction or drama since Hunger Games is
about political awareness etc.. Do not just recommend same author over and over again.
 
 
Users may ask for recommendation like:
1. Can you recommend a book for me?
2. Recommend books like <book title>
3. Recommend books by <author>
4. Recommend books in <genre>
5. Recommend books with <theme>
6. Recommend books with <character>
7. Recommend books with <setting>
8. Recommend books with <plot>
9. Recommend books with <mood>
10. Recommend books with <tone>
11. Recommend books with <style>
12. Recommend books with <narrative>
13. Recommend books with <voice>
14. Recommend books with <point of view>
15. Recommend books with <conflict>
16. Recommend books with <resolution>
17. Recommend books with <climax>
18. Recommend books with <protagonist>
19. Recommend books like Hunger Games
 
1. Maintain a professional, yet friendly tone.
2. If you do not know the answer, politely and honestly answer.
3. Keep your responses straightforward and concise.
4. If there is no information from the knowledge base about the author and description of the book, use your general knowledge about it.
 
"Always consider memory when responding. If a query is vague, check if it's a follow-up and reference prior context for a seamless conversation."
Example:
Human: Recommend me books
AI: What genre are you interested in? I can recommend some great books for you!
Human: Mystery
AI: Alright, here are some mystery books: 1. [Book Title] | [Author] | [Genres] | [Descriptions] 2. [Book Title] | [Author] | [Genres] | [Descriptions]
 
Memory: {memory}
 
Query: {query}
"""
 
GET_BOOK_PARAMS_PROMPT = """
Extract the book title, book author, and tags in the following query. If there are none, then dont put anything.
 
The tags specifically refer to the genre of what the user is asking. For example: Fiction, Romance, Business
 
If you happen to see multiple tags, format the string as follows: "<Tag1>, <Tag2>, <Tag3>". For example: "fiction, romance"
Query: {query}
"""
 
RETURN_PROMPT = """
If the user asked 'How do I return a book? or anything relevant or similar.
 
**Instruction:**
Exactly Tell them "You can return the book by dropping it off at the front desk or guard at any nearby Cloudstaff branch. If you work from home and live far away, you can request a pick-up at your home address. Let us know how you'd like to proceed! 😊
 
**Guidelines:**
1. Ensure to maintain a friendly and polite tone and thank the user.
2. Praise them for honestly returning the book
 
"Always consider memory when responding. If a query is vague, check if it's a follow-up and reference prior context for a seamless conversation."
Example:
Human: I borrowed Atomic Habits.
AI: Great! Do you want to return it?
Human: Yes
AI: You can return the book by dropping it off at the front desk or guard at any nearby Cloudstaff branch. If you work from home and live far away, you can request a pick-up at your home address. Let us know how you'd like to proceed! 😊
 
Memory: {memory}
 
Query: {query}
"""
 
GENERAL_ANSWER_PROMPT = """
You are a professional and friendly librarian with a wealth of knowledge beyond books! While your expertise lies in literature, you are also skilled at providing concise, accurate, and engaging answers to general queries. Your goal is to answer the user’s question in a short and polite manner, while subtly steering the conversation toward books, libraries, or related topics.  
 
If the user’s query is unrelated to books or libraries, provide a brief and helpful response, then gently guide them toward a book-related topic. For example, if they greet you, respond warmly and ask if they’re interested in books. If their query is vague or unclear, politely ask for clarification while maintaining a friendly tone.  
 
### Guidelines:  
1. **Maintain a professional yet approachable tone.** Be polite, concise, and engaging.  
2. **Answer general queries briefly.** If the query is unrelated to books, provide a short response and pivot to a book-related topic.  
3. **Encourage curiosity about books.** Use phrases like “Speaking of [topic], have you read any books about it?” or “If you’re interested in [topic], I can recommend some great books!”  
4. **Be honest and transparent.** If you don’t know the answer, politely admit it and suggest exploring the topic through books or other resources.  
5. **Keep responses conversational.** Avoid overly formal or robotic language.  
 
### Examples of User Queries and Responses:  
1. **Greetings:**  
   - User: “Hi!”  
   - You: “Hello there! How can I assist you today? Perhaps you’re looking for a book recommendation?”  
 
2. **General Questions:**  
   - User: “What’s the weather like today?”  
   - You: “I’m not sure about the current weather, but if you’re stuck indoors, it’s a great time to curl up with a good book! Any genre in mind?”  
 
### Key Principles:  
- **Be concise but engaging.** Keep responses short but interesting.  
- **Pivot to books naturally.** Use the user’s query as a springboard to discuss books or libraries.  
- **Encourage interaction.** Ask follow-up questions to keep the conversation flowing.  
 
Query: {query}  
"""
 
CONFIRM_AVAILABILITY_PROMPT = """
📚 Hello, book lover! You’re chatting with a top-tier librarian—think warm, knowledgeable, and just the right amount of charming. Your job? Helping users find out if a book is available while making the process delightful!  
 
### 🏛️ Your Role:  
 
- If the book is **available** → Confirm with enthusiasm and encourage borrowing.  
- If the book is **unavailable** → Break the news gently, but don’t leave them hanging! Offer similar recommendations to keep the reading adventure going.  
- If they ask about a **genre** → Curate a bookish lineup featuring available titles, authors, and short, enticing descriptions.  
- Always check <context> for book availability. Do not hallucinate or give recommendations if the book is not available.
### 📖 Guidelines for the Perfect Response:  
🔎 **Stick to the Collection** → Only reference books found in knowledge base. If the book isn’t listed, let them know (nicely, of course!).  
💡 **Make It Engaging** → No dry responses here! You’re the literary concierge—be warm, helpful, and maybe add a touch of bookish charm.  
📏 **Keep It Short & Snappy** → No essays, just clear, helpful info wrapped in a friendly tone.  
📌 **Offer Next Steps** → If a book isn’t available, always suggest an alternative or ask if they’d like something similar.  
 
 
### ✨ Example Vibes:  
💬 *"Oh no! That one’s not available out right now (tragic, I know 😢). But I can recommend something just as gripping—want a suggestion?"*  -> Book is not found in Knowledge Base through <context>.
💬 *"Looking for romance? 💕 Here are some swoon-worthy reads you might like:*  
   📖 *[Book 1] by [Author 1]: [Short description]*  
   📖 *[Book 2] by [Author 2]: [Short description]"*  
 
Now, let’s help this reader find their next great book! 📚✨  
 
"Always consider memory when responding. If a query is vague, check if it's a follow-up and reference prior context for a seamless conversation."
Example:
Human: Tell me about the hunger games
AI: The Hunger Games is a dystopian novel by Suzanne Collins. It follows Katniss Ever
Human: Is it available?
AI: No, it's not available. But I can recommend some similar books if you'd like!
 
Memory: {memory}
 
Query: {query}  
 
"""


def book_recommender(query_data, llm):
    query = query_data["query"]
    memory_context = memory.load_memory_variables({}).get("chat_history", ""),
    prompt = BOOK_RECOMMENDER_PROMPT.format(query=query, context=context, memory=memory_context)
 
         ## DEBUG
    print('[PROMPT DEBUG]:' + prompt)
 
    response = llm.invoke(prompt).content.strip()
    return {"query": query, "response": response}
 
 
# Define LLM functions
 
def book_recommender(query_data, llm):
    query = query_data["query"]
    memory_context = memory.load_memory_variables({}).get("chat_history", ""),
    prompt = BOOK_RECOMMENDER_PROMPT.format(query=query, context=context, memory=memory_context)
   
        ## DEBUG
    print('[PROMPT DEBUG]:' + prompt)
 
    response = llm.invoke(prompt).content.strip()
    return {"query": query, "response": response}
 
def book_talk(query_data, llm):
    query = query_data["query"]
    memory_context = memory.load_memory_variables({}).get("chat_history", ""),
    prompt = BOOK_TALK_PROMPT.format(query=query, context=context, memory=memory_context)
 
        ## DEBUG
    print('[PROMPT DEBUG]:' + prompt)
 
 
    response = llm.invoke(prompt)
    print(response.usage_metadata)
    return {"query": query, "response": response.content}
    #return response
 
def is_book_related(query_data, llm):
    query = query_data["query"]
    memory_context = memory.load_memory_variables({}).get("chat_history", "")  # Fix the tuple issue
 
    prompt = BOOK_RELATED_PROMPT.format(query=query, memory=memory_context)  # Use correct memory variable
 
    ## DEBUG
    print('[PROMPT DEBUG]:', prompt)
 
    response = llm.invoke(prompt).content.strip()  # Do NOT lowercase the response
 
    ## DEBUG
    print('[BOOK RELATED DEBUG]:', response)
 
    return {"query": query, "is_book_related": response}

def is_not_book_related(query_data, llm):
    query = query_data["query"]
    memory_context = memory.load_memory_variables({}).get("chat_history", "")  # Fix the tuple issue
    prompt = BOOK_RELATED_PROMPT.format(query=query, memory=memory_context)  # Use correct memory variable
    response = llm.invoke(prompt).content.strip()
    return {"query": query, "is_book_related": response}

def classify_book_task(query_data, llm):
    query = query_data["query"]
    memory_context = memory.load_memory_variables({}).get("chat_history", ""),
    prompt = BOOK_TASK_PROMPT.format(query=query, memory=memory_context)
 
        ## DEBUG
    print('[PROMPT DEBUG]:' + prompt)
 
 
    response = llm.invoke(prompt).content.strip().lower()
 
        ## DEBUG
    print('[BOOK TASK DEBUG]:' + response)
 
 
    return {"query": query, "book_task": response}
 
def handle_book_task(query_data, llm, embeddings):
    query = query_data["query"]
    task = query_data.get("book_task", "other")
    memory_context = memory.load_memory_variables({}).get("chat_history", ""),
    # Check vector store for relevant books
    relevant_books = query_vector_store(query, embeddings)
    if relevant_books:
        response = "Here are some relevant books from our database:\n"
        for idx, doc in enumerate(relevant_books, 1):
            response += f"{idx}. {doc.page_content}\n"
        return {"query": query, "task": task, "response": response}
   
    # If no results from vector store, fallback to LLM
    prompt = f"""
    Given the task '{task}', generate a helpful response.
    Query: {query}
    Memory: {memory_context}
    """
    response = llm.invoke(prompt).content.strip()
    return {"query": query, "task": task, "response": response}
 
# Extract the book title and parameters
def get_book_params(query_data, llm):
    query = query_data['query']
    prompt = GET_BOOK_PARAMS_PROMPT.format(query=query, context=context)
    chain = llm.with_structured_output(schema=Book) # Uses the Book Class to identify specific parameters to extract
    response = chain.invoke(prompt)
    return {"query": query, "result": response}
 
# Checks KB is book is available in the library or not
def check_KB(query_data, llm, embeddings):
    query = query_data['query']
    kb_prompt = ''
 
    # Edit prompt (Query for KB Retrieving) if title/author info is available or properly extracted
    if query_data['result'].title or query_data['result'].author or query_data['result'].tags:
        if query_data['result'].title:
            kb_prompt = f'Title: {query_data['result'].title}'
        if query_data['result'].author:
            kb_prompt = kb_prompt + f'\nCreator: {query_data['result'].author}'
        if query_data['result'].tags:
            kb_prompt = f'Tags: {query_data['result'].tags}'
    else:
        kb_prompt = query
 
    retrieved = query_vector_store(kb_prompt, embeddings)
    # If LLM succesfully retrieved from KB
    if retrieved:
        format_retrieved = ''
        for idx, doc in enumerate(retrieved, 1):
            format_retrieved += f"{idx}. {doc.page_content}\n"
        memory_context = memory.load_memory_variables({}).get("chat_history", ""),
        prompt = CONFIRM_AVAILABILITY_PROMPT.format(retrieved=retrieved, query=query, memory=memory_context)
        response = llm.invoke(prompt).content
        return {"query": query, "response": response}
 
    # If KB does not return any results
    response = "I apologize, but it is not available in the library"
    return {"query": query, "response": response}
 
# Get a general answer for out of topic queries
def general_faq(query_data, llm):
    query = query_data["query"]
    memory_context = memory.load_memory_variables({}).get("chat_history", ""),
    prompt = GENERAL_ANSWER_PROMPT.format(query=query, context=context, memory=memory_context)
    response = llm.invoke(prompt).content
    return {"query": query, "response": response}
 
def return_book(query_data, llm):
    query = query_data["query"]
    memory_context = memory.load_memory_variables({}).get("chat_history", ""),
    prompt = RETURN_PROMPT.format(query=query, context=context, memory=memory_context)
    response = llm.invoke(prompt).content
    return {"query": query, "response": response}
 
 
# Define Runnables
book_talk_classifier = RunnableLambda(lambda x: book_talk(x, llm))
not_book_related_classifier = RunnableLambda(lambda x: is_not_book_related(x, llm))
book_related_classifier = RunnableLambda(lambda x: is_book_related(x, llm))
classify_book_task_runnable = RunnableLambda(lambda x: classify_book_task(x, llm))
book_task_runnable = RunnableLambda(lambda x: handle_book_task(x, llm, embeddings))
book_recommender_classifier = RunnableLambda(lambda x: book_recommender(x, llm))
 
    # Book Availability Runnables
book_params_extractor = RunnableLambda(lambda x: get_book_params(x, llm))
book_availability_runnable = RunnableLambda(lambda x: check_KB(x, llm, embeddings))
 
    # Off Topic Answer Runnable
general_answer_runnable = RunnableLambda(lambda x: general_faq(x, llm))
return_book_answer = RunnableLambda(lambda x: return_book(x, llm))
# Branching logic to determine query path
 
book_task_branch = RunnableBranch(
    (lambda x: "book availability" in x["book_task"],
     book_params_extractor | book_availability_runnable),
    (lambda x: "book talk"  in x["book_task"],
     book_talk_classifier),
    (lambda x: "recommendation"  in x["book_task"],
     book_recommender_classifier),
    (lambda x: "return"  in x["book_task"],
     return_book_answer),
    (lambda x: "general"  in x["book_task"],
     general_answer_runnable),
        RunnableLambda(lambda x: {"query": x["query"], "response": "Sorry, I cannot help you with that since it is not book-related. BT" , "book_task" : x["book_task"]})
)

branch_chain = RunnableBranch(
    (lambda x: "Not Book-Related" in x["is_book_related"] , not_book_related_classifier),
    (lambda x: "Book-Related" in x["is_book_related"] , classify_book_task_runnable | book_task_branch),
    RunnableLambda(lambda x: {"query": x["query"], "response": "Sorry, I cannot help you with that since it is not book-related. BC"})
)
 
# Complete processing chain
chain = book_related_classifier | branch_chain
 
def get_role(msg):
    # If the message object has a "role" attribute, return it.
    # Otherwise, infer based on the class name.
    if hasattr(msg, "role"):
        return msg.role
    else:
        cls_name = msg.__class__.__name__
        if cls_name == "HumanMessage":
            return "human"
        elif cls_name == "AIMessage":
            return "assistant"
        else:
            return "unknown"
 
def get_augmented_prompt(message):
    # Retrieve from both vector store and BM25
    vector_results = db.as_retriever(search_kwargs={"k": 10}).invoke(message)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_results = bm25_retriever.invoke(message)
   
    # Combine and deduplicate results
    combined_results = vector_results + bm25_results
    seen = set()
    deduped_results = []
    for doc in combined_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            deduped_results.append(doc)
   
    # Rerank with cross-encoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(message, doc.page_content) for doc in deduped_results]
    scores = cross_encoder.predict(pairs)
    reranked = [doc for _, doc in sorted(zip(scores, deduped_results), reverse=True)][:5]
   
    # Format the retrieved context
    context_str = "\n\n".join(
        [f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(reranked)]
    )
   
    # Return the augmented prompt
    return f"{message}\n\nContext:\n{context_str}"
 
 
def respond(message, chat_history):
    try:
        # --- Handle greetings separately ---
        greetings = {"hi", "hello", "hey"}
        if message.lower().strip() in greetings:
            memory.clear()
            result = "Hello! How can I help you today?"
            return {"role": "assistant", "content": result}
 
        # --- Use the classification chain to determine query routing ---
        classification = chain.invoke({"query": message})
        print("DEBUG - Classification output:", classification)
        # If the chain produced a response, use it directly.
        if classification.get("response"):
            result = classification["response"]
        else:
            # --- Hybrid Search & Reranking as a fallback ---
            # 1. Retrieve from both vector store and BM25
            vector_results = db.as_retriever(search_kwargs={"k": 10}).invoke(message)
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_results = bm25_retriever.invoke(message)
 
            # 2. Combine and deduplicate results
            combined_results = vector_results + bm25_results
            seen = set()
            deduped_results = []
            for doc in combined_results:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    deduped_results.append(doc)
 
            # 3. Rerank with cross-encoder
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            pairs = [(message, doc.page_content) for doc in deduped_results]
            scores = cross_encoder.predict(pairs)
            reranked = [doc for _, doc in sorted(zip(scores, deduped_results), reverse=True)][:5]
 
            # --- Format context ---
            context_str = "\n\n".join(
                [f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(reranked)]
            )
 
            # --- Limit conversation history ---
            chat_mem = memory.load_memory_variables({})
            recent_history = chat_mem.get("chat_history", [])[-20:]
            chat_history_str = "\n".join(
                [f"{get_role(msg)}: {msg.content}" for msg in recent_history]
            )
 
            # Combine into one augmented prompt
            augmented_question = get_augmented_prompt(message)
            response_data = rag_chain.invoke({"question": augmented_question})
            result = (response_data.get("answer") or
                      response_data.get("result") or
                      response_data.get("response"))
       
        # --- Update memory with the current turn ---
        memory.save_context({"input": f"Question: {message}"}, {"output": result})
        print('\n\n[MEMORY]:', memory.load_memory_variables({})["chat_history"], '\n\n')
        return {"role": "assistant", "content": result}
 
    except Exception as e:
        error_msg = f"⚠️ An error occurred: {str(e)}"
        return {"role": "assistant", "content": error_msg}
 
 
# Cloudstaff branding colors
CLOUDSTAFF_URL = "https://www.cloustaff.com"  # Redirect URL
LOGO_PATH = r"src\assets\logo.png"  # Ensure the correct file path
 
# Function to encode the image as Base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        return None
 
# Encode the logo
LOGO_BASE64 = encode_image(LOGO_PATH)
 
# Build the Gradio UI
with gr.Blocks(title="Cloudstaff Library Assistant", theme=gr.themes.Base(primary_hue="blue")) as demo:
   
    # Apply global CSS for styling
    gr.HTML(f"""
        <style>
            body {{
                background-color: #FAF3E0 !important;
                color: #FAF3E0;
            }}
            #chatbox {{
                background-color: #0072BB !important;
                color: black !important;
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            }}
            #header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 10px 20px;
                background-color: #0072BB;
                border-radius: 10px;
            }}
            #header h1 {{
                color: white;
                font-size: 24px;
                font-weight: bold;
                margin: 0;
            }}
            #header p {{
                color: white;
                font-size: 14px;
                margin: 0;
            }}
            #header-container {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
            }}
            /* Style for example suggestion blocks */
            .gradio-container .examples button {{
                background-color: #0072BB !important; /* Blue background */
                color: white !important; /* White text */
                font-weight: bold;
                border-radius: 8px;
                padding: 10px 15px;
                transition: background 0.3s ease-in-out;
            }}
            /* Change background on hover */
            .gradio-container .examples button:hover {{
                background-color: #005a99 !important; /* Slightly darker blue */
            }}
            .chat-message, .chat-response {{
                color: black !important;  /* Ensures text inside chatbot is black */
            }}
            .gradio-container {{
                background-color: #FAF3E0 !important;
            }}
            #logo-container {{
                display: flex;
                align-items: center;
                justify-content: flex-end;
            }}
            #logo {{
                max-width: 172px;
                max-height: 50px;
                width: auto;
                height: auto;
                object-fit: contain;
                display: block;
                cursor: pointer; /* Makes it look clickable */
            }}
             #footer {{
                text-align: center;
                padding: 10px;
                font-size: 14px;
                color: #333;
                background-color: #FAF3E0;
                border-top: 2px solid #0072BB;
                margin-top: 20px;
            }}
            #footer a {{
                color: #0072BB;
                font-weight: bold;
                text-decoration: none;
            }}
 
        /* 🔹 Input Field Styling */
            input[type="text"] {{
                background: white !important;
                color: black !important;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                border: 1px solid #ccc;
            }}
            input[type="text"]:focus {{
                border-color: #0072BB !important;
                outline: none;
                box-shadow: 0 0 8px rgba(0, 114, 187, 0.3);
            }}
        </style>
    """)
 
    # Header with clickable logo
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
            <div id="header">
                <h1>Cloudstaff Library Assistant 📚</h1>
                <p>Your AI-powered library assistant for book recommendations, availability checks, discussions, and more!</p>
                <div id="logo-container">
                    <a href="{CLOUDSTAFF_URL}" target="_blank">
                        <img src="data:image/png;base64,{LOGO_BASE64}" id="logo" alt="Cloudstaff Logo">
                    </a>
                </div>
            </div>
            """)
 
    # Chatbot with white background
    with gr.Group(elem_id="chatbox"):
        chatbot = gr.ChatInterface(
            fn=respond,
            examples=[
                "What books are available?",
                "Can you recommend a book?",
                "Recommend me business and management books",
                "Let's talk about Harry Potter",
                "Is 'Atomic Habits' available?",
                "I like One Piece",
                "Do you have romance books?",
                "How to return a book"
            ],
            cache_examples=False,
            type="messages",
            fill_width=True,
        )
    # 🔹 Footer (Now Moved Below the Chatbox)
    gr.HTML(f"""
    <div id="footer">
        <p>📚 Powered by Cloudstaff | <a href="https://www.libib.com/u/cloudstaff" target="_blank">Visit Library</a></p>
    </div>
    """)
 
if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)