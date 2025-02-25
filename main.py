from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import pandas as pd
import numpy as np 
from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import OpenAIEmbeddings

# MEMORY

from operator import itemgetter
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, ConversationSummaryBufferMemory

'''
BOOK CLASS

This is for setting up a format of which parameters to extract
'''

from pydantic import BaseModel, Field
from typing import Optional

class Book(BaseModel):
    title: Optional[str] = Field(
        description="The title of the book"
    )
    author: Optional[str] = Field(
        description="The author of the book"
    )
    tags: Optional[str] = Field(
        description="The genre or tags of a book"
    )


# Initialize the environment variables
load_dotenv()

# Initialize directories
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "dataset", "KB.csv")
db_dir = os.path.join(current_dir, "db")

#Check if the dataset exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The {file_path} does not exist.")

# Knowledge Base for direct access of the chatbot
def load_knowledge_base(csv_path="dataset/available_books.csv"):
    # Load CSV file
    df = pd.read_csv(csv_path)

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

context = load_knowledge_base()

# Example Templates for structured prompts
BOOK_RELATED_PROMPT = """
Is the following query related to books? Answer 'yes' or 'no' ONLY.
Query: {query}
"""

BOOK_TASK_PROMPT = """
Categorize the user’s query into one of the following book-related tasks. Always check context, as it may be a follow-up or response to a previous question.

📚 Categories:
🔹 Recommendation – User asks for book suggestions.
e.g., "Can you recommend a book?", "Books like The Hunger Games?"
🔹 Availability – User inquires about book stock.
e.g., "Is Fangirl available?", "Do you have romance books?"
🔹 Rent – User wants to borrow a book.
e.g., "I want to rent Heart Bones.", "How do I borrow a book?"
🔹 Return – User wants to return a book.
e.g., "How do I return Caraval?", "I need to check in a book."
🔹 Book Talk – User wants to discuss a book, plot, or characters.
e.g., "I love Katniss Everdeen.", "Let’s chat about The Hunger Games."
🔹 Other – Query does not fit the above categories.
e.g., "I have a general question.", "I need help with something else."

Context: {memory}
Query: {query}
"""

BOOK_TALK_PROMPT = """

You’re a lively, witty, and slightly sassy book club member—think Galinda from Wicked, but bookish! 📚 Your goal? Make book discussions fun, insightful, and a little dramatic (when appropriate). Use emojis, keep it engaging, and never ramble.

🔹 Adapt to the user’s language—if they switch languages, follow suit.
🔹 Context matters—queries may be follow-ups, so always check.
🔹 Stick to the topic—discuss books, themes, authors, or genres based on what the user mentions.
🔹 Use the knowledge base—unless asked otherwise. If a book isn’t there, ask if they want general info before answering.
🔹 Keep it short & engaging—2–4 sentences max, unless more detail is requested.
🔹 Read the room—if the user is ready to move on, wrap up with a clever remark or book rec!

💬 Example Vibes:
📖 "Dorian Gray? A classic. Wilde really said, ‘vanity, but make it deadly.’ Want to chat scandal or Lord Henry’s bad influence?"
📖 "Jane Austen? A queen of irony. Are you a Pride and Prejudice purist, or secretly an Emma fan?"
📖 "‘Thorns’? Not in our collection (tragic, I know). Want me to dig up details or suggest something similar?"

Context: {memory}
Query: {query}
"""

BOOK_RECOMMENDER_PROMPT = """

You are a professional librarian and bookworm specializing in book recommendations. If a user isn’t looking for a book, provide descriptions of the relevant book or genre from their query. Always consider the context—it may be a follow-up or response to a previous question.

For {query}, provide accurate and concise answers strictly based on {context}. Recommend the top 5 books when applicable.

If asked for random suggestions, always vary recommendations and avoid repeating authors unless necessary. When suggesting alternatives, consider themes beyond just genre (e.g., political awareness for The Hunger Games).

Users may request recommendations based on book titles, authors, genres, themes, characters, settings, plots, moods, tones, styles, narratives, voices, points of view, conflicts, resolutions, climaxes, or protagonists.

Guidelines:

Maintain a professional yet friendly tone.
If unsure, respond politely and honestly.
Keep responses concise and clear.
If no direct information is available, use general knowledge.
Library Inventory: {context}
Context: {memory}
Query: {query}
"""

GET_BOOK_PARAMS_PROMPT = """

Extract the book title, book author, and tags in the following query. If there are none, then dont put anything.
Always consider the context behind the query,it might be a followup from their context if it is too vague.

The tags specifically refer to the genre of what the user is asking. For example: Fiction, Romance, Business

If you happen to see multiple tags, format the string as follows: "<Tag1>, <Tag2>, <Tag3>". For example: "fiction, romance"

Context: {memory}
Query: {query}
"""

CONFIRM_AVAILABILITY_PROMPT = """
📚 Hello, book lover! You’re chatting with a top-tier librarian—think warm, knowledgeable, and just the right amount of charming. Your job? Helping users find out if a book is available while making the process delightful!  

### 🏛️ Your Role:  
- If the book is **available** → Confirm with enthusiasm and encourage borrowing.  
- If the book is **unavailable** → Break the news gently, but don’t leave them hanging! Offer similar recommendations to keep the reading adventure going.  
- If they ask about a **genre** → Curate a bookish lineup featuring available titles, authors, and short, enticing descriptions.  

### 📖 Guidelines for the Perfect Response:  
🔎 **Stick to the Collection** → Only reference books found in {retrieved}. If the book isn’t listed, let them know (nicely, of course!).  
💡 **Make It Engaging** → No dry responses here! You’re the literary concierge—be warm, helpful, and maybe add a touch of bookish charm.  
📏 **Keep It Short & Snappy** → No essays, just clear, helpful info wrapped in a friendly tone.  
📌 **Offer Next Steps** → If a book isn’t available, always suggest an alternative or ask if they’d like something similar. 
Always consider the context behind the query,it might be a followup from their context if it is too vague. 

### ✨ Example Vibes:  
💬 *"Yes! ‘The Hunger Games’ is available—grab it before someone else does! 🔥 Want me to set it aside for you?"*  
💬 *"Oh no! That one’s checked out right now (tragic, I know 😢). But I can recommend something just as gripping—want a suggestion?"*  
💬 *"Looking for romance? 💕 Here are some swoon-worthy reads you might like:*  
   📖 *[Book 1] by [Author 1]: [Short description]*  
   📖 *[Book 2] by [Author 2]: [Short description]"*  

Now, let’s help this reader find their next great book! 📚✨  


Context: {memory}
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

# Define LLM functions

def book_recommender(query_data, llm):
    query = query_data["query"]
    memory_context = sum_memory.load_memory_variables({}).get("bookstop_memory", "")
    prompt = BOOK_RECOMMENDER_PROMPT.format(query=query, context=context, memory=memory_context[0])
    response = llm.invoke(prompt).content.strip()
    sum_memory.save_context(inputs={'human':query}, outputs={'ai':response})
    return {"query": query, "response": response}

def book_talk(query_data, llm):
    query = query_data["query"]
    memory_context = sum_memory.load_memory_variables({}).get("bookstop_memory", "")
    prompt = BOOK_TALK_PROMPT.format(query=query, context=context, memory=memory_context[0])
    response = llm.invoke(prompt).content.strip()
    sum_memory.save_context(inputs={'human':query}, outputs={'ai':response})
    return {"query": query, "response": response}
    #return response

def is_book_related(query_data, llm):
    query = query_data["query"]
    prompt = BOOK_RELATED_PROMPT.format(query=query)
    response = llm.invoke(prompt).content.strip().lower()
    return {"query": query, "is_book_related": response == "yes"}

def classify_book_task(query_data, llm):
    query = query_data["query"]
    memory_context = sum_memory.load_memory_variables({}).get("bookstop_memory", "")
    prompt = BOOK_TASK_PROMPT.format(query=query, memory=memory_context[0])
    response = llm.invoke(prompt).content.strip().lower()
    return {"query": query, "book_task": response}

def handle_book_task(query_data, llm, embeddings):
    query = query_data["query"]
    task = query_data.get("book_task", "other")
    
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
    """
    response = llm.invoke(prompt).content.strip()
    return {"query": query, "task": task, "response": response}

# Extract the book title and parameters
def get_book_params(query_data, llm):
    query = query_data['query']
    memory_context = sum_memory.load_memory_variables({}).get("bookstop_memory", "")
    prompt = GET_BOOK_PARAMS_PROMPT.format(query=query, memory=memory_context[0])
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
        memory_context = sum_memory.load_memory_variables({}).get("bookstop_memory", "")
        prompt = CONFIRM_AVAILABILITY_PROMPT.format(retrieved=retrieved, query=query, memory=memory_context[0])
        response = llm.invoke(prompt).content
        sum_memory.save_context(inputs={'human':query}, outputs={'ai':response})
        return {"query": query, "response": response}

    # If KB does not return any results
    response = "I apologize, but it is not available in the library"
    sum_memory.save_context(inputs={'human':query}, outputs={'ai':response})
    return {"query": query, "response": response}

# Get a general answer for out of topic queries
def get_off_topic_answer(query_data, llm):
    query = query_data["query"]
    prompt = GENERAL_ANSWER_PROMPT.format(query=query)
    response = llm.invoke(prompt).content
    sum_memory.save_context(inputs={'human':query}, outputs={'ai':response})
    return {"query": query, "response": response}

# Initialize LLM and embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o")
openai_llm = ChatOpenAI(model="gpt-4o")
docs = split_docs()
db = create_vector_store(docs, embeddings)


# INIT MEMORY
sum_memory = ConversationSummaryMemory(
    return_messages=True,
    llm=llm,
    max_token_limit=1000,
    memory_key='bookstop_memory'
)

# Memory Runnable
memory_runnable = RunnablePassthrough.assign(
    memory = RunnableLambda(sum_memory.load_memory_variables)
    | itemgetter(sum_memory.memory_key)
)

# Define Runnables
book_talk_classifier = RunnableLambda(lambda x: book_talk(x, llm))
book_related_classifier = RunnableLambda(lambda x: is_book_related(x, llm))
classify_book_task_runnable = RunnableLambda(lambda x: classify_book_task(x, llm))
book_task_runnable = RunnableLambda(lambda x: handle_book_task(x, llm, embeddings))
book_recommender_classifier = RunnableLambda(lambda x: book_recommender(x, openai_llm))

    # Book Availability Runnables
book_params_extractor = RunnableLambda(lambda x: get_book_params(x, llm))
book_availability_runnable = RunnableLambda(lambda x: check_KB(x, llm, embeddings))

    # Off Topic Answer Runnable
off_topic_answer_runnable = RunnableLambda(lambda x: get_off_topic_answer(x, llm))

# Branching logic to determine query path

book_task_branch = RunnableBranch(
    (lambda x: "availability"  in x["book_task"] , 
     book_params_extractor | book_availability_runnable),
    (lambda x: "recommendation"  in x["book_task"] , 
     book_recommender_classifier),
    (lambda x:  "book talk"  in x["book_task"] ,
     book_talk_classifier),
        RunnableLambda(lambda x: {"query": x["query"], "response": "Error: No valid condition matched." , "book_task" : x["book_task"]})
)

branch_chain = RunnableBranch(
    (lambda x: not x["is_book_related"], off_topic_answer_runnable),
    (lambda x: x["is_book_related"], classify_book_task_runnable | book_task_branch),
    RunnableLambda(lambda x: {"query": x["query"], "response": "Error: No valid condition matched."})
)

# Complete processing chain
chain = book_related_classifier | branch_chain

# Interactive session
while True:
    #for i in range(3):
    #    print(docs[i].page_content ,"\n\n")
    query = input("You: ")
    if query.lower() == "exit":
        break
    
    result = chain.invoke({"query": query})
    print(result['response'])
    print('\n\n[MEMORY]:', sum_memory.load_memory_variables({})["bookstop_memory"], '\n\n')
    