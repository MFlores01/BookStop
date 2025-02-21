# BookStop Library Assistant 📚
# Developers
1. ![Miguel Flores]([[images/miguel.png](https://media.licdn.com/dms/image/v2/D5603AQHzCsQys79VXw/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1726156627180?e=1745452800&v=beta&t=YZbNNR3fOkvsLyxGg3_zIidWilmhbqTWh2qY3gvZY3k)](https://media.licdn.com/dms/image/v2/D5603AQHzCsQys79VXw/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1726156627180?e=1745452800&v=beta&t=YZbNNR3fOkvsLyxGg3_zIidWilmhbqTWh2qY3gvZY3k)) Miguel Flores  
2. ![Andrei Miranda]([images/andrei.png](https://media.licdn.com/dms/image/v2/D5603AQH7F4ctmE9Vcg/profile-displayphoto-shrink_400_400/B56ZSMY_imGQAg-/0/1737522149949?e=1745452800&v=beta&t=l-n8yD4el1plvq_N_J3yfm-VSish__x_NKTQ--DaKmw)) Andrei Miranda  
3. ![Gabrielle Torres]([images/gabrielle.png](https://media.licdn.com/dms/image/v2/D5603AQE1dk5DfdGHug/profile-displayphoto-shrink_800_800/B56ZR0tOKpHsAg-/0/1737124800105?e=1745452800&v=beta&t=82Nzf2Nej7Ev19RdWMgkwTQsv6v0H5ylr7_-PXL42BY)) Gabrielle Torres  
4. ![Kentz Chavez]([images/kentz.png](https://media.licdn.com/dms/image/v2/D5603AQEqJ_N052vPSg/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1726317334829?e=1745452800&v=beta&t=3TQjzNn318tLUFTQWAe-JEFKkvedkcASmOe3VcUHjxk)) Kentz Chavez  
5. ![Gregory Samson]([[images/gregory.png](https://media.licdn.com/dms/image/v2/D4E03AQGohXQjlBAPdw/profile-displayphoto-shrink_800_800/B4EZS6HK1eHUAg-/0/1738289229573?e=1745452800&v=beta&t=sTO3K041rQd0DkC8uS7uz_RpoHBQ-4BDdeMzfOQTbas)](https://media.licdn.com/dms/image/v2/D4E03AQGohXQjlBAPdw/profile-displayphoto-shrink_800_800/B4EZS6HK1eHUAg-/0/1738289229573?e=1745452800&v=beta&t=sTO3K041rQd0DkC8uS7uz_RpoHBQ-4BDdeMzfOQTbas)) Gregory Samson
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
- **Vector Database:** The embeddings are stored in `db`.
- **Environment Variables:** Load OpenAI or Google API keys via `.env`.
 
## Troubleshooting
- **FileNotFoundError for CSV**:
  - Ensure `dataset` exists and is correctly formatted.
 
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
