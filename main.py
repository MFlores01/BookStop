import sys
import os

# Ensure Python finds 'src/' by adding it to sys.path
sys.path.insert(0, os.path.abspath("src"))

# Now import modules correctly
from src.core.vector_store import VectorStore
from src.core.chatbot import BookChatbot
from src.core.gradio_ui import create_gradio_ui
from src.core.prompt_templates import PromptTemplates
from dotenv import load_dotenv

load_dotenv()
def main():
    """Main entry point for the Cloudstaff Library Assistant."""
    chatbot = BookChatbot()  # Initialize chatbot

    # Start the Gradio UI
    ui = create_gradio_ui()
    ui.launch()

if __name__ == "__main__":
    main()
