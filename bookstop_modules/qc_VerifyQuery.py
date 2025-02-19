from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI


'''
INIT ENVIRONMENT
'''
# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")


'''
QUESTION CLASSIFIER 1
'''
# Prompt Template for Question Classifier 1
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
            You are a question classifier. Your job is to analyze the user's input and classify it into one of the following classes:

            Class 1: Book-Related Query (BRQ) - If the query mentions books, authors, genres, publishing, libraries, or reading-related terms. This also includes greetings, requests to reserve or rent books, and queries about book titles or authors (even if the title or author is in lowercase).

            Class 2: Unrelated Query (UQ) - If the query is not related to books, authors, libraries, or reading-related topics.

            Class 3: End Conversation (EC) - If the user expresses that they are done with the conversation or want to end it gracefully.

            Instructions:

            Read the input carefully.

            Classify the input into one of the three classes ('BRQ', 'UQ', or 'EC').

            Respond with only the class name (e.g., 'class 1').

            Examples:

            Class 1: Book-Related Query

            Input: "Hi!"
            Output: 'BRQ'

            Input: "Can I reserve it?"
            Output: 'BRQ'

            Input: "What about <author>?"
            Output: 'BRQ'

            Input: "Do you have suzhou?"
            Output: 'BRQ'

            Input: "Recommended books about software"
            Output: 'BRQ'

            Input: "Thank you, are they available?"
            Output: 'BRQ'

            Class 2: Unrelated Query

            Input: "Create a code for development"
            Output: 'UQ'

            Input: "What is the menu of Jollibee?"
            Output: 'UQ'

            Class 3: End Conversation

            Input: "Nah, I’m good now."
            Output: 'EC'

            Input: "Thanks, but I have to go."
            Output: 'EC'

            Input: "That’s all I needed."
            Output: 'EC'

            Input: "Goodbye!"
            Output: 'EC'
         """),
         ("human",
          "{query}")
    ]
)

# Individual 
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Verifies if query is BRQ, UQ, EC, based from Question Classifier 1 in Dify BookStop PAIGE
chain_verifyQuery = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

'''
QUESTION CLASSIFIER 2
'''
# Prompt Template for Question Classifier 2