# BookStop Library Assistant 📚
# Developers
1. Miguel Flores
2. Andrei Miranda
3. Gabrielle Torres
4. Kentz Chavez
5. Gregory Samson

## Overview
BookStop is an AI-powered library assistant that provides book recommendations, availability checks, and engaging book discussions. Built using LangChain, OpenAI embeddings, and Gradio, this chatbot makes it easy to explore and interact with books in a dynamic way.
 
## Features
- 📖 **Book Recommendations**: Get personalized book suggestions based on genres, authors, or themes.
- 🔍 **Book Availability Check**: Quickly check if a book is available in the library’s collection.
- 🗣️ **Book Discussions**: Chat about books, authors, and literary themes.
- 🤖 **Intelligent Query Handling**: Uses an LLM to classify queries as book-related or general.
- 📂 **Vector Database**: Stores and retrieves book information efficiently.
- 🏛 **User-Friendly Interface**: Built with Gradio for a seamless experience.
 
## Technologies Used
- **LangChain**: Manages LLM interactions and query classification.
- **OpenAI & Google Generative AI**: Provides AI-powered responses.
- **ChromaDB**: Stores vector embeddings for efficient retrieval.
- **Gradio**: Creates an interactive web-based chat UI.
- **Pandas & NumPy**: Handles dataset operations.
 
## Installation
### Prerequisites
- Python 3.8+
- Virtual Environment (Recommended)
 
### Steps
1. **Clone the Repository**
   ```sh
   git clone https://github.com/MFlores01/BookStop
   cd BookStop
   ```
2. **Create and Activate a Virtual Environment**
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```
3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the Application**
   ```sh
   python try.py
   ```
 
## Usage
- Start the chatbot using Gradio.
- Type queries like:
  - *"Recommend a book similar to The Hunger Games"*
  - *"Is Pride and Prejudice available?"*
  - *"Who is the author of The Daily Stoic?"*
 
## Configuration
- **Dataset Path:** Update `file_path` in `try.py` to point to your book dataset.
- **Vector Database:** The embeddings are stored in `db/KB_db`.
- **Environment Variables:** Load OpenAI or Google API keys via `.env`.
 
## Troubleshooting
- **Invalid color error in Gradio theme**:
  - Replace `secondary_hue="brown"` with `secondary_hue="orange"` or remove it.
- **FileNotFoundError for CSV**:
  - Ensure `dataset/KB.csv` exists and is correctly formatted.
 
## Contributing
- Fork the repository.
- Create a new branch: `git checkout -b feature-name`
- Commit changes: `git commit -m "Add new feature"`
- Push changes: `git push origin feature-name`
- Open a pull request.
 
## License
This project is licensed under the MIT License. See `LICENSE` for details.
 
## Acknowledgments
- Inspired by book enthusiasts and library lovers in Cloudstaff.
- Built using open-source AI and data science tools.
 
---
📚 *Happy reading!*
