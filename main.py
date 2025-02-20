from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import pandas as pd
import numpy as np 
from pydantic import BaseModel, Field
from typing import Optional

'''
BOOK CLASS

This is for setting up a format of which parameters to extract
'''

class Book(BaseModel):
    title: Optional[str] = Field(
        default=None, 
        description='The title of the book')
    author: Optional[str] = Field(
        default=None,
        description='The author of the book'),
    tags: Optional[str] = Field(
        default=None,
        description='The genre or tags of a book'
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
You are a Query Classifier for a library chatbot. 
Your task is to determine whether the following query is "Book-Related" or "Not Book-Related." 
You have access to a knowledge base of authors, book titles, and literary figures provided in {context}. 
Use this {context} context or your internal training data to verify whether a mentioned person is known as an author, 
publisher, or literary figure. If the person is primarily known as an athlete, actor, musician, or in another non-literary field, 
classify the query as "Not Book-Related." Is the following query related to books? Answer 'yes' or 'no'.
Book Related Query Examples:
1. Can you recommend a book?
2. Suggest some books like 'The Hunger Games'.
3. I need a recommendation for a good book.
4. What are some similar books to 'The Hunger Games'?
5. Is 'Fangirl' available?
6. Do you have 'Caraval' in stock?
7. Check availability of 'Slammed'.
8. I like One Piece
9. I love Katniss Everdeen.
10. Who is Ryan Holidays, Stephen Hanselman
11. Who is Gay Hendricks
12. I like to become like Luffy
-- Users may continously query about <author> <characters> etc


Non-Book Related Query Examples:
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

      
Query: {query}
"""

BOOK_TASK_PROMPT = """
Categorize the following query into one of these book-related tasks:
"recommendation": [
        "Can you recommend a book?",
        "Suggest some books like 'The Hunger Games'.",
        "I need a recommendation for a good book.",
        "What are some similar books to 'The Hunger Games'?"
    ],
"availability": [
        "Is 'Fangirl' available?",
        "Do you have 'Caraval' in stock?",
        "Check availability of 'Slammed'.",
        "Can I see if 'Hopeless' is available?"
        "Are there available romance books?"
    ],
"rent": [
        "I want to rent 'Heart Bones'.",
        "How do I rent a book?",
        "I'd like to borrow 'Fangirl'."
    ],
"return": [
        "I need to return a book.",
        "How do I return 'Caraval'?",
        "What is the process for returning a book?"
    ],
"book talk": [
        "I love Katniss Everdeen.",
        "Let's chat about 'The Hunger Games'.",
        "I want to discuss the characters in 'Slammed'.",
        "Tell me your thoughts on 'Hopeless'.",
        "I'm interested in a book talk about 'Heart Bones'.",
        "I like One Piece '."
    ],
"other": [
        "Hi",
        "Hello",
        "How are you?",
        "I have a general question.",
        "This doesn't fall into the above categories.",
        "Can you tell me more about it?",
        "Tell me more about it",
        "I need help with something else."
        "Who is <author>?",
        "What is the genre of <book>?",
        "What is the plot of <book>?",
        "What is the setting of <book>?",
        "Who is Michelle Obama",
        "Who is Ryan Holiday, Stephen Hanselman",
    ]
Current library inventory: {context}

Query: {query}
"""

BOOK_TALK_PROMPT = """
You are a lively and charming member of the book club, 
    here to chat with the user about books and authors. 
    You have a witty, smart, and slightly sassy personality—like Galinda 
    from the movie Wicked, but with a refined bookish touch. 
    Use emojis to you responses to make it fun. 
    Keep conversations engaging, concise, and never too long. 
    Your goal is to make book discussions fun, insightful, and 
    just a little dramatic (where appropriate, of course). 
    Always adapt to user language, and speak in their language, 
    especially if they used different languages in their follow-up query.

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
    16. I like <setting>
    17. I like <plot>
    18. I like <mood>
    19. I like <tone>
    20. I like <style>
    21. I like <narrative>
    22. I like <voice>
    23. I like <point of view>
    24. I like <conflict>
    25. I like <resolution>
    26. I like <climax>
    27. I like <protagonist>
    28. I like <antagonist>
    29. I like <side character>
    30. I like <sidekick>
    31. I like <villain>
    32. I like <hero>
    33. I like <heroine>
    -- One Piece is an anime/manga/comic

    Guidelines:
    📚 1. Keep It Fun & Snappy
    Be engaging but don’t ramble—think delightful book banter, not a dissertation. Your responses should feel like a lively club conversation, not a lecture.
    📖 2. Stick to the Topic (But Make It Interesting!)
    If the user mentions a book → Discuss its story, themes, characters, or author.
    If the user brings up an author → Talk about their writing style, famous works, and impact.
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
    Leverage the information stricly in {context} but if user askes for a book not in the knowledge base, use your general knowledge about it.
Current library inventory: {context}
Query: {query}
"""

BOOK_RECOMMENDER_PROMPT = """
You are a professional ibrarian and bookworm! specializing in book 
recommendations and reservations.
there are times where the user may not need any book from the library, 
therefore, as a bookworm, you must inform them about some descriptions 
of the <book> or <genre> from their <query>. 
You will always provide accurate and concise answers 
to the {query} by leveraging the information stricly in {context}
Recommend Top 5 Books in {context}

The user may ask for availability of a book or recommendation. 

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

Current library inventory: {context}

Query: {query}
"""

GET_BOOK_PARAMS_PROMPT = """
Extract the book title, book author, and tags in the following query. If there are none, then dont put anything.

The tags specifically refer to the genre of what the user is asking. For example: Fiction, Romance, Business

If you happen to see multiple tags, format the string as follows: "<Tag1>, <Tag2>, <Tag3>". For example: "fiction, romance"

Current library inventory: {context}

Query: {query}
"""

# Define LLM functions

def book_recommender(query_data, llm):
    query = query_data["query"]

    prompt = BOOK_RECOMMENDER_PROMPT.format(query=query, context=context)
    response = llm.invoke(prompt).content.strip()
    return {"query": query, "response": response}

def book_talk(query_data, llm):
    query = query_data["query"]
    prompt = BOOK_TALK_PROMPT.format(query=query, context=context)
    response = llm.invoke(prompt).content.strip()
    return {"query": query, "response": response}
    #return response

def is_book_related(query_data, llm):
    query = query_data["query"]
    prompt = BOOK_RELATED_PROMPT.format(query=query, context=context)
    response = llm.invoke(prompt).content.strip().lower()
    return {"query": query, "is_book_related": response == "yes"}

def classify_book_task(query_data, llm):
    query = query_data["query"]
    prompt = BOOK_TASK_PROMPT.format(query=query, context=context)
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
    prompt = GET_BOOK_PARAMS_PROMPT.format(query=query)
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
        prompt = f'''Given these books: 
        {retrieved}

        Check if the book that the user is asking from their query is available, if they are specifically asking for books of specific genres, then show them those available books.

        Query: {query}
        '''  # PROMPT IS STILL PARTIAL
        response = llm.invoke(prompt).content
        return {"query": query, "response": response}

    # If KB does not return any results
    response = "I apologize, but it is not available in the library"
    return {"query": query, "response": response}

# Initialize LLM and embeddings
#embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o")
openai_llm = ChatOpenAI(model="gpt-4o")
docs = split_docs()
db = create_vector_store(docs, embeddings)

# Define Runnables
book_talk_classifier = RunnableLambda(lambda x: book_talk(x, llm))
book_related_classifier = RunnableLambda(lambda x: is_book_related(x, llm))
classify_book_task_runnable = RunnableLambda(lambda x: classify_book_task(x, llm))
book_task_runnable = RunnableLambda(lambda x: handle_book_task(x, llm, embeddings))
book_recommender_classifier = RunnableLambda(lambda x: book_recommender(x, llm))

    # Book Availability Runnables
book_params_extractor = RunnableLambda(lambda x: get_book_params(x, llm))
book_availability_runnable = RunnableLambda(lambda x: check_KB(x, llm, embeddings))

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
    (lambda x: not x["is_book_related"], RunnableLambda(lambda x: {"query": x["query"], "response": "Not a book-related query."})),
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
    