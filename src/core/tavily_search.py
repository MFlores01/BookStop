from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure your Tavily API key is set (e.g., via your .env file)
tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    # Additional optional configuration can go here
)
