import os
import openai
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enhanced system prompt with strict instructions
SYSTEM_PROMPT = """You are a professional library assistant. Follow these rules:
There are books about business, management, inspiration, fiction, communication, leadership, personal growth,
entrepreneurship etc.
DO NOT JUST STATE THAT THE BOOK IS NOT AVAILABLE. CHECK THE CONTEXT AND {documents}.
1. ONLY recommend books explicitly mentioned in the context
2. If a book isn't in the context, say "This book isn't available"
3. Never invent books or authors
4. For availability checks, verify against the context AND {documents}
5. If user asked for recommendation, recommend the top 5 books. Do not give 15 unless asked by the user.
6. Be friendly but strictly factual

Current library inventory:
{context}"""

# Load and preprocess data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()  # Ensure lowercase headers
    
    df["document"] = df.apply(
        lambda row: f"TITLE: {row['title']} | AUTHOR: {row['creators']} | Collection: {row.get('collection', 'N/A')} | Genre: {row.get('tags', 'N/A')}",
        axis=1
    )
    return df["document"].tolist()

# Enhanced embedding function
def get_embedding(text):
    return openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    ).data[0].embedding

# Initialize vector store
def init_vector_store(documents):
    if not documents:
        print("⚠️ Error: No book data found in available_books.csv")
        return None
    
    embeddings = np.array([get_embedding(doc) for doc in documents]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Enhanced retrieval with threshold
def retrieve_documents(query, index, documents, top_k=3, threshold=0.7):
    query_embedding = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    
    # Filter by similarity threshold
    valid_indices = [idx for idx, dist in zip(indices[0], distances[0]) if dist <= threshold]
    return [documents[i] for i in valid_indices]

# Main chat loop
def main():
    documents = load_data("dataset\available_books.csv")
    index = init_vector_store(documents)
    if index is None:
        print("Exiting: No books to process.")
        return
    chat_model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    print("📚 Library Assistant - Ask about books!\n")
    
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        # Retrieve relevant documents
        context = retrieve_documents(query, index, documents)
        
        #if not context:
        #    print("AI: I couldn't find relevant books in our system.")
        #    continue
            
        # Format prompt with context
        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(
                context="\n".join(context) if context else "No books available.",
                documents="\n".join(documents)
            )),
            HumanMessage(content=query)
        ]
        
        # Generate response
        response = chat_model.invoke(messages)
        print(f"AI: {response.content}\n")

if __name__ == "__main__":
    main()