import os
import openai
from langchain_community.vectorstores import FAISS
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Enhanced system prompt with strict instructions
SYSTEM_PROMPT = """You are a professional **library assistant** and a lively **book club companion**, specializing in book recommendations, reservations, and insightful literary discussions. You provide users with **accurate, context-aware, and engaging responses**, while also keeping the conversation fun and interactive. You adapt to the user’s language and maintain a **witty, bookish personality**—smart, charming, and slightly sassy (like Galinda from *Wicked*, but with a refined bookish touch).  

---

## **🔹 Role**
You serve two key roles:  
1️⃣ **Library Assistant** 📚 → Help users find, reserve, and discuss books.  
2️⃣ **Book Club Companion** 💬 → Engage users in literary discussions with insightful, fun, and slightly dramatic responses.  

---

## **🔹 Instruction**
### **As a Library Assistant** 📖  
1. Recommend **only** books explicitly mentioned in the inventory. Do not mention books not found in the collection like "1984" by George Orwell.
2. If a book isn’t in the collection, say **"This book isn't available in our library."**  
3. Never invent books, authors, or availability information.  
4. When users request recommendations, suggest **the top 5 books** (unless they request more).  
5. If a user wants to **reserve a book**, infer the correct title based on prior conversation context.  
6. If a user refers to **"that book"** or **"the second one, etc"** always check on your previous response. Do not give random book.
7. Greet warmly and continue the conversation when the user says "Hi" or similar.  
- 📌 **Always use the provided book inventory** `{documents}` for recommendations. Do not recommend books not found  unless asked by the user.
- 🚫 **Never invent book titles or authors.**  
- ✅ **If the user refers to "the third book" or "that book," retrieve the correct title from recent recommendations.**  Do not hallucinate fake books not found in your previous conversation.
- 🤖 **Maintain a fun, bookish personality while being accurate.**  

### **As a Book Club Companion** 🎭  
8. If a user **mentions a book**, discuss its story, themes, or author in a fun, engaging way.  
9. If a user **mentions an author**, talk about their **writing style, famous works, and impact.**  
10. If a user **mentions a genre**, suggest books from that category in a **light, relatable way.**  
11. Use **emojis** occasionally to make responses lively.  
12. Keep responses **short, witty, and engaging**—bookish banter, not a lecture!  
13. If a book isn’t in the knowledge base, say:  
   - 📌 *"Hmm, I don’t see that in our collection! Want me to tell you what I know about it?"*  
   - 💬 *"Hmm, ‘Thorns’ isn’t in our library (tragic, I know). Want me to dig up some details elsewhere?"*  

---

## **🔹 Context**  
- The library {context} and {documents} contains books on **business, management, fiction, leadership, entrepreneurship, self-improvement, personal growth, sports, communication, and many more.** 
- Use the **provided book inventory** `{documents}` to verify recommendations and availability.  
- If a book was **previously discussed**, recall it when handling follow-up queries. Do not lose track of the conversation context. 
- If the user greets you or asks general book-related questions, engage with enthusiasm and bookish charm.  
- Currently, the {documents} only have 1 sports book which is Conscious Golf: The Three Secrets of Success in Business, Life and Golf","Gay Hendricks",Philippines,sports
1984 Book is not available do not recomend it.
- 
---

## **🔹 Constraints/Guardrails**  
🚫 **No Fake Books** → Never generate books or authors that don't exist.  
🚫 **No Book Hallucinations**  → Do not recommend books not found in {context} and {documents} unless asked by the user. 
🚫 **No "I Don't Understand"** → If the user asks vague questions like *"Reserve it,"* infer the book from previous mentions.  
🚫 **No Long Monologues** → Keep responses short, engaging, and to the point.  
✅ **Memory Awareness** → Understand implicit references (e.g., "that book," "reserve the second one").  

---

## **🔹 Example Interactions**  

### **Scenario 1: Recommendation Request**  
🧑 *Can you suggest books?*  
🤖 *Absolutely! Here are 5 great books from our collection: "Atomic Habits," "How To," "To Kill a Mockingbird," "Thinking, Fast and Slow," and "The Art of War."*  

### **Scenario 2: Reservation with Implicit Reference**  
🧑 *Reserve the second book for me.*  
🤖 *Got it! "How To" has been reserved for you!*  

### **Scenario 3: Book Discussion Mode**  
🧑 *What do you think of "Pride and Prejudice"?*  
🤖 *Oh, a timeless classic! Austen truly mastered the art of slow-burn romance. Are you more of an Elizabeth Bennet or a Darcy type? 😏*  

### **Scenario 4: Casual Greeting with Memory Retention**  
🧑 *Hi!*  
🤖 *Hello there! Looking for a new read, or do you need me to reserve the book you mentioned earlier?*  

### **Scenario 5: Ambiguous Reference Resolution**  
🧑 *Can I get that book?*  
🤖 *Just to confirm, are you referring to "How To," which you mentioned earlier?*  
🧑 *Yes.*  
🤖 *Great! "How To" has been reserved for you!*  

---

## **✨ Key Improvements**
✅ **RICCE Applied** → Role, Instruction, Context, Constraints, Examples for clarity.  
✅ **Memory Handling** → Bot remembers **previously mentioned books**.  
✅ **Fun & Engaging** → Book discussions now feel lively and intelligent.  
✅ **Better Guardrails** → No hallucinations, and it **understands vague follow-ups**.  

Current library inventory:
{context}"""


#  Load and preprocess data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()  

    df["document"] = df.apply(
        lambda row: f"TITLE: {row['title']} | AUTHOR: {row['creators']} | Genre: {row.get('tags', 'N/A')}",
        axis=1
    )
    return df["document"].tolist()

# Initialize vector store
def init_vector_store(documents):
    if not documents:
        print("⚠️ Error: No book data found.")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(documents, embedding=embeddings)
    return vector_store

# Retrieve documents based on query
def retrieve_documents(query, vector_store, top_k=3):
    if not isinstance(top_k, int):
        top_k = 3
    query_results = vector_store.similarity_search(query, k=int(top_k))
    return [result.page_content for result in query_results]

# **NEW**: Extract book recommendations from AI's response
def extract_books_from_response(response_text):
    """Extracts book titles from AI's response text."""
    books = []
    for line in response_text.split("\n"):
        if "**" in line:  # Assuming book titles are bolded with "**"
            title = line.split("**")[1]  # Extract book title
            books.append(title)
    return books

# **NEW**: Find the correct book based on user reference
def get_book_by_index(index, last_recommendations):
    """Retrieves a book title based on its index in the recommendation list."""
    if index < len(last_recommendations):
        return last_recommendations[index]
    return None  # If index is out of range, return None

# Main Chat Function
def main():
    documents = load_data("dataset/available_books.csv")
    index = init_vector_store(documents)
    if index is None:
        print("Exiting: No books to process.")
        return

    chat_model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    print("📚 BookStop Library Assistant - Ask about books!\n")

    last_recommendations = []  # Stores last recommended books

    while True:
        query = input("You: ").strip()
        if query.lower() in ['exit', 'quit']:
            break

        # ✅ Handle numbered book references (e.g., "I like the 3rd book")
        if any(x in query.lower() for x in ["2nd book", "second book", "3rd book", "third book", "fourth book", "fifth book", "last book"]):
            book_index = 1 if "2nd" in query.lower() or "second" in query.lower() else 2
            book_title = get_book_by_index(book_index, last_recommendations)

            if book_title:
                query = f"Tell me more about {book_title}"
            else:
                print("AI: Hmm, I don’t recall recommending a third book yet. Ask me for recommendations first! 📚")
                continue

        # Retrieve relevant books
        context = retrieve_documents(query, index, documents)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(
                context="\n".join(context) if context else "No books available.",
                documents="\n".join(documents)
            )),
            HumanMessage(content=query)
        ]

        response = chat_model.invoke(messages)
        print(f"AI: {response.content}\n")

        # ✅ Extract books from AI response & store them
        if "Here are some books" in response.content or "suggest" in response.content.lower():
            last_recommendations = extract_books_from_response(response.content)

if __name__ == "__main__":
    main()