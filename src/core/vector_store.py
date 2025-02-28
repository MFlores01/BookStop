import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize directories
current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
file_path = os.path.join(current_dir, "src", "dataset", "KB.csv")
db_dir = os.path.join(current_dir, "db")

# Check if the dataset exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at '{file_path}'")

class VectorStore:
    """Handles the vector storage and retrieval operations."""
    def __init__(self, embedding_model="text-embedding-3-small"):
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.docs = self._split_docs()
        self.db = self._create_vector_store()
        self.bm25_retriever = self._create_bm25_retriever()
        self.hybrid_retriever = self._create_ensemble_retriever()

    def _split_docs(self):
        """Loads and splits documents from CSV"""
        loader = CSVLoader(file_path=file_path)  # ✅ Fixed
        documents = loader.load()
        splitter = CharacterTextSplitter(separator="\n", chunk_size=2000)
        return splitter.split_documents(documents)
    
    def _create_vector_store(self):
        """Creates or loads an existing Chroma vector store."""
        persistent_directory = os.path.join(db_dir, "KB_db")
        
        if not os.path.exists(persistent_directory):
            print("\n-- Initializing the vector database ---")
            db = Chroma.from_documents(self.docs, self.embeddings, persist_directory=persistent_directory)
            print("--- Finished Creating Vector Store ---")
        else:
            print("\n--- Retrieving the existing vector database ---")
            db = Chroma(persist_directory=persistent_directory, embedding_function=self.embeddings)
        
        return db  # ✅ Ensure it returns the created/retrieved DB
        
    def _create_bm25_retriever(self):
        """Creates a BM25 retriever for the vector store."""
        retriever = BM25Retriever.from_documents(self.docs)
        retriever.k = 10  # Retrieve top 10 documents
        return retriever

    def _create_ensemble_retriever(self):
        """Creates an ensemble retriever combining BM25 and vector retrieval."""
        vector_retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        return EnsembleRetriever(retrievers=[self.bm25_retriever, vector_retriever], weights=[0.5, 0.5])
    
    def query_vector_store(self, query):
        """Retrieves relevant documents from the vector store"""
        retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        return retriever.invoke(query)
    
    def retrieve_hybrid_results(self, query):
        """Retrieves results using Hybrid Retrieval"""
        return self.hybrid_retriever.invoke(query)
