from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings


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
        description='The author of the book'
    )



# PROMPT TEMPLATES ============


GET_BOOK_PARAMS_PROMPT = """
Extract the book title or book author in the following query. If there are none, then dont put anything.
Example: The Hunger Games,
Query: {query}
"""

RETRIEVE_BOOK_PROMPT = """

"""

# LLM FUNCTIONS
def get_book_params(query_data, llm):
    query = query_data['query']
    prompt = GET_BOOK_PARAMS_PROMPT.format(query=query)
    chain = llm.with_structured_output(schema=Book)
    response = chain.invoke(prompt) 
    return {"query": query, "result": response }

def check_KB(query_data, llm, embeddings):
    query = query_data['query']
    kb_prompt = ''

    # Edit prompt if title/author info is available or properly extracted
    if query_data['result'].title or query_data['result'].author:
        if query_data['result'].title:
            kb_prompt = f'Title: {query_data['result'].title}'
        if query_data['result'].author:
            kb_prompt = kb_prompt + f'\nCreator: {query_data['result'].author}'
    else:
        kb_prompt = query

    retrieved = query_vector_store(kb_prompt, embeddings)
    # If LLM retrieved from KB
    if retrieved:
        format_retrieved = ''
        for idx, doc in enumerate(retrieved, 1):
            format_retrieved += f"{idx}. {doc.page_content}\n"
        prompt = f'''Given these books: 
        {retrieved}

        Check if the book that the user is asking from their query is available

        Query: {query}
        '''
        response = llm.invoke(prompt).content
        return {"query": query, "response": response}

    # Book is not available
    response = "I apologize, but it is not available in the library"
    return {"query": query, "response": response}
    


# Initialize LLM and embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o")

docs = split_docs()
db = create_vector_store(docs, embeddings)

# DEfine Runnables
extract_params = RunnableLambda(lambda x: get_book_params(x, llm))
check_availability = RunnableLambda(lambda x: check_KB(x, llm, embeddings))


'''
TEST
'''

chain = extract_params | check_availability

# Interactive session
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    
    result = chain.invoke({"query": query})

    print('[][] RESPONSE:', result['response'])