from langchain_community.document_loaders import CSVLoader  
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
import gradio as gr
from pydantic import BaseModel, Field
from typing import Optional
from core.prompt_templates import PromptTemplates
from core.vector_store import VectorStore

vector_store = VectorStore()
PromptTemplates = PromptTemplates()

class Book(BaseModel):
    """Defines the book parameters to extract from queries."""
    title: Optional[str] = Field(None, description="The title of the book")
    author: Optional[str] = Field(None, description="The author of the book")
    tags: Optional[str] = Field(None, description="The genre or tags of a book")


class BookChatbot:
    """Handles book-related chatbot functionalities using LangChain pipelines."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.memory = ConversationBufferMemory()
        self.embeddings = OpenAIEmbeddings()

    def fetch_memory_context(self):
        """Fetches chat history from memory."""
        return self.memory.load_memory_variables({}).get("chat_history", "")
    
    def is_book_related(self, query_data: dict) -> dict:
        """Classifies if a query is book-related or not."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context()
        prompt = PromptTemplates.book_related_prompt(query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content.strip().lower()
        # DEBUG: Log the response
        print('[BOOK RELATED DEBUG]:', response)
        return {"query": query, "is_book_related": response}
    
    def book_talk(self, query_data):
        """Engages in discussion about books, summaries, or analysis."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context()
        prompt = PromptTemplates.book_talk_prompt(query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content.strip()
        return {"query": query, "response": response}

    def book_recommender(self, query_data):
        """Handles book recommendation queries."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context()
        prompt = PromptTemplates.book_recommendation_prompt(query=query, memory=memory_context)
        #print('[PROMPT DEBUG]:', prompt)  # Debugging log
        response = self.llm.invoke(prompt).content.strip()
        return {"query": query, "response": response}

    def classify_book_task(self, query_data):
        """Classifies the user's book-related query into specific tasks."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context()
        prompt = PromptTemplates.book_task_prompt(query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content.strip().lower()
        #print('[BOOK TASK DEBUG]:', response)  # Debugging log
        return {"query": query, "book_task": response}

    def get_book_params(self, query_data):
        """Extracts structured book parameters (title, author, tags) from a query."""
        query = query_data['query']
        prompt = PromptTemplates.get_book_params_prompt(query=query)
        chain = self.llm.with_structured_output(schema=Book)
        response = chain.invoke(prompt)
        return {"query": query, "result": response}

    def general(self, query_data):
        """Handles general book-related queries."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context()
        prompt = PromptTemplates.general_answer_prompt(query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content
        return {"query": query, "response": response}

    def book_return(self, query_data):
        """Handles book return queries."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context()
        prompt = PromptTemplates.return_prompt(query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content
        return {"query": query, "response": response}

    def check_KB(self, query_data):
        """Checks if a book is available in the knowledge base."""
        query = query_data['query']
        book_details = query_data.get('result', Book())

        # Construct KB query
        details = [
            f'Title: {book_details.title}' if book_details.title else "",
            f'Creator: {book_details.author}' if book_details.author else "",
            f'Tags: {book_details.tags}' if book_details.tags else ""
        ]
        kb_prompt = "\n".join(filter(None, details)) or query  # Default to query if no details

        retrieved = self.vector_store.query_vector_store(kb_prompt)
        
        if retrieved:
            formatted_response = "\n".join([f"{idx+1}. {doc.page_content}" for idx, doc in enumerate(retrieved)])
            memory_context = self.fetch_memory_context()
            prompt = PromptTemplates.confirm_availability(retrieved=formatted_response, query=query, memory=memory_context)
            response = self.llm.invoke(prompt).content
            return {"query": query, "response": response}
        
        return {"query": query, "response": "I apologize, but it is not available in the library."}


    def book_task_pipeline(self):
            """Defines the main chain pipeline for book-related tasks."""

            # Convert functions into LangChain RunnableLambda
            book_params_extractor = RunnableLambda(lambda x: self.get_book_params(x))
            book_availability_runnable = RunnableLambda(lambda x: self.check_KB(x))

            book_task_branch = RunnableBranch(
                (lambda x: "Book Availability" in x["book_task"], RunnableLambda(self.get_book_params) | RunnableLambda(self.check_KB)),
                (lambda x: "Book Recommendation" in x["book_task"], RunnableLambda(self.book_recommender)),
                (lambda x: "Book Talk" in x["book_task"], RunnableLambda(self.book_talk)),
                (lambda x: "Book Return" in x["book_task"], RunnableLambda(self.book_return)),
                (lambda x: "General" in x["book_task"], RunnableLambda(self.general)),
                RunnableLambda(lambda x: {"query": x["query"], "response": "Sorry, I cannot help with that."})
            )

            return RunnableLambda(lambda x: self.classify_book_task(x)) | book_task_branch

    def main_pipeline(self):
        """Defines the primary chatbot query handling pipeline."""
        branch_chain = RunnableBranch(
            (lambda x: "Book-Related" in x["is_book_related"], self.book_task_pipeline()),
            (lambda x: "Not Book-Related" in x["is_book_related"], RunnableLambda(lambda x: {"query": x["query"], "response": "Sorry, this is not book-related."})),
            RunnableLambda(lambda x: {"query": x["query"], "response": "Unknown query type."})
        )

        return RunnableLambda(self.is_book_related) | branch_chain


# Initialize chatbot instance
chatbot = BookChatbot()
