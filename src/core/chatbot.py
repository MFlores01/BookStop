from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json
#from sentence_transformers import CrossEncoder
import os
import gradio as gr
import logging
from pydantic import BaseModel, Field
from typing import Optional
from core.prompt_templates import PromptTemplates
from core.prompt_templates2 import Template  # Import the new prompt templates
from core.vector_store import VectorStore
from core.tavily_search import tavily_tool
from core.class_vocabulary import ClassVocab
from langchain_core.output_parsers import StrOutputParser
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore
from datetime import datetime
import asyncio
logging.basicConfig(level=logging.DEBUG)
 
 
 
vector_store = VectorStore()
PromptTemplates = PromptTemplates()
 
 
class Book(BaseModel):
    """Defines the book parameters to extract from queries."""
    title: Optional[str] = Field(None, description="The title of the book")
    author: Optional[str] = Field(None, description="The author of the book")
    tags: Optional[str] = Field(None, description="The genre or tags of a book")
 
 
class BookChatbot:
    """Handles book-related chatbot functionalities using LangChain pipelines."""
 
    def __init__(self, session_id="user1_session", max_messages=5):
        self.llm = ChatOpenAI(model="gpt-4o")
 
        # Initialize Firestore connection
        self.firestore_client = firestore.Client()
        self.memory = FirestoreChatMessageHistory(
            session_id=session_id,
            collection="chat_history",  # Firestore collection for chat history
            client=self.firestore_client
        )
 
        self.max_messages = max_messages
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = VectorStore()
        self.templates = Template()
 
        # Load messages into memory
        stored_messages = self.memory.messages
        self.stored_messages = self.memory.messages[-self.max_messages:] if stored_messages else []
 
        #self.stored_messages = self.memory.messages[-self.max_messages:]  # Store last few messages
 
    def get_chat_history(self, session_id="user1_session"):
        """Fetches the conversation history from Firestore."""
        session_ref = self.firestore_client.collection("chat_history").document(session_id)
        session_doc = session_ref.get()
 
        if session_doc.exists:
            return session_doc.to_dict().get("messages", [])
        return []
 
    def save_chat_history(self, user_query: str, bot_response: str):
        """Appends chat messages to the user's session history in Firestore."""
        timestamp = datetime.utcnow().isoformat()  # Ensure timestamp is ISO formatted string
 
        # Store messages as JSON strings
        user_message = json.dumps({
            "role": "user",
            "content": user_query,
            "timestamp": timestamp
        })
       
        bot_message = json.dumps({
            "role": "bot",
            "content": bot_response,
            "timestamp": timestamp
        })
 
        self.memory.add_message(HumanMessage(content=user_message))
        self.memory.add_message(AIMessage(content=bot_message))    
 
        print("[Chat History Updated]:", user_query, "->", bot_response)
 
 
    async def fetch_memory_context(self, query_data: dict = None):
        """Fetches chat history from Firestore and ensures it's formatted as a list of messages."""
       
        stored_messages = await self.memory.aget_messages()  # ✅ Properly await async function
 
        if not stored_messages:  # Ensure stored_messages is not None
            print("⚠️ No messages retrieved from memory.")
            return []
 
        print(f"✅ Retrieved {len(stored_messages)} messages.")  # Debugging
        stored_messages = stored_messages[-self.max_messages:]  # Slice last N messages
 
        formatted_messages = []
        for msg in stored_messages:
            print("🔍 Raw message:", msg)  # Debugging
 
            try:
                msg_content = json.loads(msg.content)  # Decode JSON
                print("✅ Decoded message:", msg_content)  # Debugging
 
                if msg_content.get("role") == "user":
                    formatted_messages.append(HumanMessage(content=msg_content["content"]))
                else:
                    formatted_messages.append(AIMessage(content=msg_content["content"]))
            except json.JSONDecodeError as e:
                print(f"⚠️ Error decoding message: {msg.content} | {e}")  # Debugging
                continue
 
        return formatted_messages  # ✅ Returns a list of messages
 
   
    def not_related(self, query_data):
        """Handles queries that are not book-related."""
        query = query_data["query"].strip().lower()
        memory_context = self.fetch_memory_context(query_data)
        prompt = self.templates.not_related_template().format(chat_history=memory_context, query=query)
        response = self.llm.invoke(prompt).content
        return {"query": query, "response": response}
 
    def is_book_related(self, query_data: dict) -> dict:
        """Classifies if a query is book-related or not."""
        query = query_data["query"].strip().lower()
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
        if any(word in query for word in ClassVocab.book_recommendation_vocab()):
            rule_task = "book recommendation"
        elif any(word in query for word in ClassVocab.book_availability_vocab()):
            rule_task = "book availability"
        elif any(word in query for word in ClassVocab.book_rent_vocab()):
            rule_task = "book rent"
        elif any(word in query for word in ClassVocab.book_return_vocab()):
            rule_task = "book return"
        elif any(word in query for word in ClassVocab.book_talk_vocab()):
            rule_task = "book talk"
        elif any(word in query for word in ClassVocab.general_vocab()):
            rule_task = "general"
        elif any(word in query for word in ClassVocab.not_book_related_vocab()):
            rule_task = "not book-related"
        else:
            rule_task = "other"
       
        # LLM-based classification fallback
            # If rule-based classifier is uncertain ("other"), use LLM-based few-shot classification.
        if rule_task == "other":
            memory_context = self.fetch_memory_context(query_data)
            prompt = self.templates.book_task_template().format(chat_history=memory_context, query=query)
            llm_task = self.llm.invoke(prompt).content.strip().lower()
            if llm_task not in ["book recommendation", "book talk", "book availability", "book rent", "book return", "general", "not book-related"]:
                llm_task = "other"
            final_task = llm_task
        else:
            final_task = rule_task
        
        #input_tokens = llm_task.usage_metadata.get("input_tokens", 0) if hasattr(llm_task, "usage_metadata") else 0
        #output_tokens = llm_task.usage_metadata.get("output_tokens", 0) if hasattr(llm_task, "usage_metadata") else 0
        #print(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}")  # Debugging
        print(f"[HYBRID CLASSIFICATION DEBUG] Query: '{query}' | Rule: '{rule_task}' | Final: '{final_task}'")
        return {"query": query, "book_task": final_task}
 
    def check_KB(self, query_data):
        """Checks if a book is available in the knowledge base."""
        query = query_data["query"].strip().lower()
 
        memory_context = asyncio.run(self.fetch_memory_context(query_data))
 
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
            prompt = self.templates.book_availability_template().format(
                context=formatted_response, chat_history=memory_context, query=query
            )
            result = self.llm.invoke(prompt)
            input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
            output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0
            print(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}")  # Debugging
            response = result.content
            return {"query": query, "response": response}
 
        return {"query": query, "response": "I'm sorry, but I couldn't find any relevant information in my knowledge base."}
 
       
    def book_rent(self, query_data):
        """Handles book rental queries."""
        query = query_data["query"].strip().lower()
 
        memory_context = asyncio.run(self.fetch_memory_context(query_data))
 
        prompt = self.templates.book_rent_template().format(chat_history=memory_context, query=query)
        result = self.llm.invoke(prompt)
        input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
        output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0
        print(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}")  # Debugging
        response = result.content
        return {"query": query, "response": response}
 
 
    def book_talk(self, query_data):
        """Engages in discussion about books, summaries, or analysis."""
        query = query_data["query"].strip().lower()
       
        # ✅ Properly call async function inside a sync function
        memory_context = asyncio.run(self.fetch_memory_context(query_data))  
 
        book_details = query_data.get('result', Book())
 
        # Retrieve relevant information from the knowledge base
        kb_prompt = f"Title: {book_details.title}, Author: {book_details.author}, Tags: {book_details.tags}" if book_details.title or book_details.author else query
        retrieved = self.vector_store.query_vector_store(kb_prompt)
 
        if retrieved:
            formatted_response = "\n".join([f"{idx+1}. {doc.page_content}" for idx, doc in enumerate(retrieved)])
            prompt = self.templates.book_talk_template().format(context=formatted_response, chat_history=memory_context, query=query)
            result = self.llm.invoke(prompt)
            input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
            output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0
            print(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}")  # Debugging
            response = result.content
            return {"query": query, "response": response}
 
        return {"query": query, "response": "I'm sorry, I couldn't find any relevant book discussion in my knowledge base."}
 
 
    def book_recommender(self, query_data):
        """Handles book recommendation queries."""
        query = query_data["query"].strip().lower()
       
        memory_context = asyncio.run(self.fetch_memory_context(query_data))
 
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
            prompt = self.templates.book_recommend_template().format(
                context=formatted_response, chat_history=memory_context, query=query
            )
            result = self.llm.invoke(prompt)
            input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
            output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0
            print(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}")  # Debugging
            response = result.content
            return {"query": query, "response": response}
       
        return {"query": query, "response": "I'm sorry, but I couldn't find any recommendations available in the library for you."}
 
    def general(self, query_data):
        """Handles general book-related queries."""
        query = query_data["query"].strip().lower()
 
        memory_context = asyncio.run(self.fetch_memory_context(query_data))
 
        # Retrieve information from vector store
        retrieved = self.vector_store.query_vector_store(query)
 
        if retrieved:
            formatted_response = "\n".join([f"{idx+1}. {doc.page_content}" for idx, doc in enumerate(retrieved)])
            prompt = self.templates.general_answer_template().format(context=formatted_response, chat_history=memory_context, query=query)
            result = self.llm.invoke(prompt)
            input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
            output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0
            print(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}")  # Debugging
            response = result.content
            return {"query": query, "response": response}
 
        return {"query": query, "response": "I don't have an answer for that right now, but feel free to ask something else!"}
 
 
    def book_return(self, query_data):
        """Handles book return queries."""
        query = query_data["query"].strip().lower()
       
        memory_context = asyncio.run(self.fetch_memory_context(query_data))
 
        # Retrieve book-related info from the vector store
        retrieved = self.vector_store.query_vector_store(query)
 
        if retrieved:
            formatted_response = "\n".join([f"{idx+1}. {doc.page_content}" for idx, doc in enumerate(retrieved)])
            prompt = self.templates.book_return_template().format(context=formatted_response, chat_history=memory_context, query=query)
            result = self.llm.invoke(prompt)
            input_tokens = result.usage_metadata.get("input_tokens", 0) if hasattr(result, "usage_metadata") else 0
            output_tokens = result.usage_metadata.get("output_tokens", 0) if hasattr(result, "usage_metadata") else 0
            print(f"Tokens Used - Input: {input_tokens}, Output: {output_tokens}")  # Debugging
            response = result.content
            return {"query": query, "response": response}
 
        return {"query": query, "response": "I couldn't find book return details in my records. Please check with the library."}
 
    def check_history(self,query_data):
        query = query_data["query"]
        prompt = self.templates.history_template()
 
        chain = prompt | self.model
        result = chain.invoke({"query": query, "chat_history":self.get_recent_messages()})
 
        logging.debug(f"History: {result}")  
 
        return {
            "response": result.content
        }  
   
    def book_task_pipeline(self, query: str):
        """Defines the main chain pipeline for book-related tasks while retaining chat history."""
       
        # Retrieve past chat history for context
        chat_history = self.get_chat_history()  
 
        # Prepare the query data with chat history
        query_data = {
            "query": query,
            "chat_history": chat_history  # Pass full conversation history
        }
       
        # Classify the user's query into a specific book-related task
        task_result = self.classify_book_task(query_data)
 
        if task_result.get("book_task") == "book fallback":
            # If no specific task is identified, fetch data online
            tavily_result = tavily_tool.invoke({"query": query})
            response = f"Here's what I found online:\n{tavily_result.content}"
        else:
            # Define the branching logic for book-related tasks
            book_task_branch = RunnableBranch(
                (lambda x: "book availability" in x["book_task"], RunnableLambda(lambda x: self.check_KB(x))),
                (lambda x: "book recommendation" in x["book_task"], RunnableLambda(lambda x: self.book_recommender(x))),
                (lambda x: "book talk" in x["book_task"], RunnableLambda(lambda x: self.book_talk(x))),
                (lambda x: "book return" in x["book_task"], RunnableLambda(lambda x: self.book_return(x))),
                (lambda x: "general" in x["book_task"], RunnableLambda(lambda x: self.general(x))),
                (lambda x: "book rent" in x["book_task"], RunnableLambda(lambda x: self.book_rent(x))),
                (lambda x: "not book-related" in x["book_task"], RunnableLambda(lambda x: self.not_related(x))),
                RunnableLambda(lambda x: {"query": x["query"], "response": "Sorry, I cannot help with that."})
            )
 
            # Invoke the correct branch based on task classification
            response_data = book_task_branch.invoke(task_result)
            response = response_data["response"]
 
        # Save the conversation history to Firestore
        self.save_chat_history(query, response)
       
        # Return final response
        return {"query": query, "response": response}
 
 
 
# Initialize chatbot instance
chatbot = BookChatbot()
 
 
 