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
#from sentence_transformers import CrossEncoder
import os
import gradio as gr
from pydantic import BaseModel, Field
from typing import Optional
from core.prompt_templates import PromptTemplates
from core.vector_store import VectorStore
from core.tavily_search import tavily_tool
 
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
        self.vector_store = VectorStore()
 
    def fetch_memory_context(self, query_data: dict = None) -> str:
        """Fetches chat history from memory and optionally combines it with provided chat_history."""
        stored = self.memory.load_memory_variables({}).get("chat_history", "")
        if query_data and "chat_history" in query_data and query_data["chat_history"]:
            extra = query_data["chat_history"]
            if isinstance(extra, list):
                extra_str_list = []
                for item in extra:
                    if isinstance(item, dict):
                        # Get the 'content' field if available; otherwise convert the whole dict to string.
                        value = item.get("content")
                        if value is None:
                            value = str(item)
                        else:
                            value = str(value)
                        extra_str_list.append(value)
                    else:
                        extra_str_list.append(str(item))
                extra = "\n".join(extra_str_list)
            else:
                extra = str(extra)
            return stored + "\n" + extra
        return stored

    
    def not_related(self, query_data):
        """Handles queries that are not book-related."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context(query_data)
        prompt = PromptTemplates.not_related_prompt(query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content
        return {"query": query, "response": response}

    def is_book_related(self, query_data: dict) -> dict:
        """Classifies if a query is book-related or not."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context(query_data)
        prompt = PromptTemplates.book_related_prompt(query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content.strip().lower()
        # DEBUG: Log the response
        print('[BOOK RELATED DEBUG]:', response)
        return {"query": query, "is_book_related": response}

    def classify_book_task(self, query_data):
        """Hybrid classification for book-related tasks.
        
        Uses both rule-based classification and LLM-based fallback. If the rule-based classifier returns 'other',
        then the LLM output is used.
        """
        query = query_data["query"].strip().lower()
        if not query:
            return {"error": "Query cannot be empty.", "book_task": "other", "chat_history": self.fetch_memory_context(query_data)}
        
        # Rule-based classification
        if any(word in query for word in ["AI", "medical", "coding", "sport", "give me a book", "Find", "recommend", "suggest", "good book", "any book", "kids", "novel", "books for kids"]):
            rule_task = "book recommendation"
        elif any(word in query for word in ["available", "in stock", "do you have", "availability"]):
            rule_task = "book availability"
        elif any(word in query for word in ["to avail", "take", "rent", "borrow", "lend", "can i get", "check out", "issue book"]):
            rule_task = "book rent"
        elif any(word in query for word in ["how to return a book", "return", "give back", "bring back", "returning", "drop off"]):
            rule_task = "book return"
        elif any(word in query for word in ["glaze", "fire", "tea", "emotion", "disgusting", "dont like", "not so much", "has", "pretty", "meh", "hate", "Talk", "know", "learn", "amazed", "love", "like", "chat", "think", "analyze", "thoughts", "discussion", "talk about", "opinions", "review"]):
            rule_task = "book talk"
        elif any(word in query for word in ["como esta", "slay", "skrrt", "skibidi", "hi", "hello", "hey", "how are you", "good morning", "good afternoon", "good evening"]):
            rule_task = "general"
        elif any(word in query for word in ["salary", "pay", "work", "job", "employee", "staff", "cloudstaff", "address", "lloyd", "joy", "password", "code", "python"]):
            rule_task = "not book-related"
        else:
            rule_task = "other"
        
        # LLM-based classification fallback
            # If rule-based classifier is uncertain ("other"), use LLM-based few-shot classification.
        if rule_task == "other":
            memory_context = self.fetch_memory_context(query_data)
            prompt = PromptTemplates.book_task_prompt(query=query, memory=memory_context)
            llm_task = self.llm.invoke(prompt).content.strip().lower()
            if llm_task not in ["book recommendation", "book talk", "book availability", "book rent", "book return", "general", "not book-related"]:
                llm_task = "other"
            final_task = llm_task
        else:
            final_task = rule_task

        print(f"[HYBRID CLASSIFICATION DEBUG] Query: '{query}' | Rule: '{rule_task}' | Final: '{final_task}'")
        return {"query": query, "book_task": final_task}
    
    def get_book_params(self, query_data):
        """Extracts structured book parameters (title, author, tags) from a query."""
        query = query_data['query']
        prompt = PromptTemplates.get_book_params_prompt(query=query)
        chain = self.llm.with_structured_output(schema=Book)
        response = chain.invoke(prompt)
        return {"query": query, "result": response}

 
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

        # Dynamic fallback: If no documents are retrieved or retrieval is weak
        # Then consider this a fallback case
        if not retrieved:
            return {"query": query, "book_task": "book fallback"}

        # Otherwise, proceed with normal availability check
        formatted_response = "\n".join([f"{idx+1}. {doc.page_content}" for idx, doc in enumerate(retrieved)])
        memory_context = self.fetch_memory_context(query_data)
        prompt = PromptTemplates.confirm_availability(retrieved=formatted_response, query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content
        return {"query": query, "response": response}
       
    def book_rent(self, query_data):
        """Handles book rental queries."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context(query_data)
        prompt = PromptTemplates.book_rent_prompt(query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content
        return {"query": query, "response": response}

    def book_talk(self, query_data):
        """Engages in discussion about books, summaries, or analysis."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context(query_data)
        prompt = PromptTemplates.book_talk_prompt(query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content.strip()
        return {"query": query, "response": response}
 
    def book_recommender(self, query_data):
        """Handles book recommendation queries."""
        query = query_data["query"]
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
            memory_context = self.fetch_memory_context(query_data)
            prompt = PromptTemplates.book_recommendation_prompt(retrieved=formatted_response, query=query, memory=memory_context)
            response = self.llm.invoke(prompt).content
            return {"query": query, "response": response}
       
        return {"query": query, "response": "I'm sorry, but I couldn't find any recommendations available in the library for you."}

    def general(self, query_data):
        """Handles general book-related queries."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context(query_data)
        prompt = PromptTemplates.general_answer_prompt(query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content
        return {"query": query, "response": response}
 
    def book_return(self, query_data):
        """Handles book return queries."""
        query = query_data["query"]
        memory_context = self.fetch_memory_context(query_data)
        prompt = PromptTemplates.return_prompt(query=query, memory=memory_context)
        response = self.llm.invoke(prompt).content
        return {"query": query, "response": response}


    def book_task_pipeline(self, query: str, chat_history: str):
        """Defines the main chain pipeline for book-related tasks."""
        # Create a dictionary that includes both query and chat_history
        query_data = {"query": query, "chat_history": chat_history}

        # First, classify the task using hybrid rule-based and LLM fallback classifier
        task_result = self.classify_book_task(query_data)

        # If classification returns a fallback indicator (e.g., "book fallback"), then use Tavily search
        if task_result.get("book_task") == "book fallback":
            tavily_result = tavily_tool.invoke({"query": query})
            return {"query": query, "response": f"Here's what I found online:\n{tavily_result.content}"}

        # Convert functions into LangChain RunnableLambda
        book_params_extractor = RunnableLambda(lambda x: self.get_book_params(x))
        book_availability_runnable = RunnableLambda(lambda x: self.check_KB(x))
        book_talk_runnable = RunnableLambda(lambda x: self.book_talk(x))
        book_rent_runnable = RunnableLambda(lambda x: self.book_rent(x))
        book_recommender_runnable = RunnableLambda(lambda x: self.book_recommender(x))
        return_runnable = RunnableLambda(lambda x: self.book_return(x))
        general_runnable = RunnableLambda(lambda x: self.general(x))
        not_related_runnable = RunnableLambda(lambda x: self.not_related(x))

        # Build the branch with a new condition for "not book-related"
        book_task_branch = RunnableBranch(
            (lambda x: "book availability" in x["book_task"], book_params_extractor | book_availability_runnable),
            (lambda x: "book recommendation" in x["book_task"], book_recommender_runnable),
            (lambda x: "book talk" in x["book_task"], book_talk_runnable),
            (lambda x: "book return" in x["book_task"], return_runnable),
            (lambda x: "general" in x["book_task"], general_runnable),
            (lambda x: "book rent" in x["book_task"], book_rent_runnable),
            (lambda x: "not book-related" in x["book_task"], not_related_runnable),  # <-- New branch for not book-related queries
            RunnableLambda(lambda x: {"query": x["query"], "response": "Sorry, I cannot help with that."})
        )

        # Chain classification with the branch.
        # We pass the entire query_data (which includes chat_history) to classify_book_task.
        book_related_chain = RunnableLambda(lambda x: self.classify_book_task(x))
        chain = book_related_chain | book_task_branch
        result = chain.invoke(query_data)
        return result


# Initialize chatbot instance
chatbot = BookChatbot()
 