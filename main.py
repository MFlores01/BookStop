import sys
import os
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory

# Ensure Python finds 'src/' by adding it to sys.path
sys.path.insert(0, os.path.abspath("src"))

# Now import modules correctly
from src.core.chatbot import BookChatbot
from src.core.gradio_ui import create_gradio_ui
from dotenv import load_dotenv

load_dotenv()

# Setup Firebase Firestore
PROJECT_ID = "langchaincs"
SESSION_ID = "user1_session"  # This could be a username or a unique ID
COLLECTION_NAME = "chat_history"

# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# FirestoreChatMessageHistory is already being used inside BookChatbot,
# so we just need to pass the session_id instead of manually passing chat_history.
def main():
    """Main entry point for the Cloudstaff Library Assistant."""
    chatbot = BookChatbot(session_id=SESSION_ID)  # ✅ Pass session_id, not memory

    # Start the Gradio UI
    ui = create_gradio_ui()
    ui.launch()

if __name__ == "__main__":
    main()
