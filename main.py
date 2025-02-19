from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

# Initialize the environment variables
load_dotenv()

# Initialize directories
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "datasets", "available_books.csv")
db_dir = os.path.join(current_dir, "db")

#Check if the dataset exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The {file_path} does not exist.")

# Load and split documents
def split_docs():
    loader = CSVLoader(file_path, encoding="utf-8")
    documents = loader.load()
    splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=500)
    return splitter.split_documents(documents)

# Create or retrieve vector store
def create_vector_store(docs, embeddings):
    persistent_directory = os.path.join(db_dir, "available_book_db")
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
    persistent_directory = os.path.join(db_dir, "available_book_db")
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

# Example Templates for structured prompts
BOOK_RELATED_PROMPT = """
Is the following query related to books? Answer 'yes' or 'no'.
Query: {query}
"""

BOOK_TASK_PROMPT = """
Categorize the following query into one of these book-related tasks:
- recommendation
- availability
- rent
- return
- other (if none of the above)

Query: {query}
"""

# Define LLM functions
def is_book_related(query_data, llm):
    query = query_data["query"]
    prompt = BOOK_RELATED_PROMPT.format(query=query)
    response = llm.invoke(prompt).content.strip().lower()
    return {"query": query, "is_book_related": response == "yes"}

def classify_book_task(query_data, llm):
    query = query_data["query"]
    prompt = BOOK_TASK_PROMPT.format(query=query)
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

# Initialize LLM and embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

docs = split_docs()
db = create_vector_store(docs, embeddings)

# Define Runnables
book_related_classifier = RunnableLambda(lambda x: is_book_related(x, llm))
classify_book_task_runnable = RunnableLambda(lambda x: classify_book_task(x, llm))
book_task_runnable = RunnableLambda(lambda x: handle_book_task(x, llm, embeddings))

# Branching logic to determine query path
branch_chain = RunnableBranch(
    (lambda x: not x["is_book_related"], RunnableLambda(lambda x: {"query": x["query"], "response": "Not a book-related query."})),
    (lambda x: x["is_book_related"], classify_book_task_runnable | book_task_runnable),
    RunnableLambda(lambda x: {"query": x["query"], "response": "Error: No valid condition matched."})
)

# Complete processing chain
chain = book_related_classifier | branch_chain

# Interactive session
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    
    result = chain.invoke({"query": query})
    print(result)
    print(result['response'])