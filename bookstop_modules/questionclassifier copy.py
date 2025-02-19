from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableBranch
from langchain_openai import ChatOpenAI

'''
INIT ENVIRONMENT
'''
# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")


'''
QUESTION CLASSIFIER 1 - verifyQuery

This is for classifying if user's query is valid or not
'''
# Prompt Template for Question Classifier 1
prompt_template1 = ChatPromptTemplate.from_messages(
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

# Verifies if query is BRQ, UQ, EC, based from Question Classifier 1 in Dify BookStop PAIGE
# BRQ - Book Related Query
# UQ - Unrelated Query
# EC - End Conversation
chain_verifyQuery = prompt_template1 | model 


'''
QUESTION CLASSIFIER 2 - classifyQuery

This is for classifying which classification does user's query belong
'''

# Prompt Template for Question Classifier 2
prompt_template2 = ChatPromptTemplate.from_messages([
    ("system",
     """
    You are a question classifier. Your job is to analyze the user's input and classify it into one of the following classes:

    Specific Book Availability (SBA) - If the user asks whether a specific book is available to borrow in the library.

    Book Recommendation (BR) - If the user asks for a book recommendation (by genre, author, or similar books).

    Book Description Inquiry (BDI) - If the user asks about the details of a specific book (synopsis, genre, author, etc.).

    Book Talk (BT) - If the user wants to chat about a specific book or author.

    Book Renting/Borrowing (BB) - If the user wants to rent or borrow a specific book.

    Book Returning/Checking in (BC) - If the user wants to return or check in a book.

    Instructions:

    Read the input carefully.

    Consider the user's previous dialogues for context.

    Classify the input into one of the six classes ('SBA', 'BR', 'BDI', 'BT', 'BB', or 'BC').

    Respond with only the class abbreviation (e.g., 'SBA').

    Examples:

    Specific Book Availability (SBA):

    Input: "Is The Hunger Games available?"
    Output: 'SBA'

    Input: "Do you have Atomic Habits?"
    Output: 'SBA'

    Book Recommendation (BR):

    Input: "Can you recommend me a fantasy book?"
    Output: 'BR'

    Input: "I'm thinking of a book about fantasy, fiction, and anime."
    Output: 'BR'

    Book Description Inquiry (BDI):

    Input: "What is the synopsis of The Hunger Games?"
    Output: 'BDI'

    Input: "What genre is Atomic Habits?"
    Output: 'BDI'

    Book Talk (BT):

    Input: "Let's talk about The Hunger Games!"
    Output: 'BT'

    Input: "I just finished reading Atomic Habits."
    Output: 'BT'

    Book Renting/Borrowing (BB):

    Input: "Can I rent The Hobbit?"
    Output: 'BB'

    Input: "I want to borrow Atomic Habits."
    Output: 'BB'

    Book Returning/Checking in (BC):

    Input: "I want to return Atomic Habits."
    Output: 'BC'

    Input: "Can I return The Hobbit?"
    Output: 'BC'
    """),
    ("human",
     "{query}"
     )
])

# Individual
format_prompt2 = RunnableLambda(lambda x: prompt_template2.format_prompt(**x))

# Classifies where book query belongs:
# SBA - Specific Book Availability
# BR - Book Recommendation
# BDI - Book Description Inquiry
# BT - Book Talk
# BB - Book Renting/BORROWING
# BC - Book Returning/CHECKING In

chain_classifyQuery = prompt_template2 | model 


'''
BRANCH - query Classification Branch
'''
qc_Branch = RunnableBranch(
    # Branch 1: Book-Related Query (BRQ)
    (
        lambda x: "BRQ" in x,
        RunnableLambda(lambda x: (print("DEBUG: Passing to classifyQuery:", x), chain_classifyQuery))  # Print and return result
    ),
    (
        lambda x: "UQ" in x,
        RunnableLambda(lambda x: (print("DEBUG: Unrelated Query:", x), "I'm sorry, let us not go beyond the topic of books and this library!")[1])
    ),
    (
        lambda x: "EC" in x,
        RunnableLambda(lambda x: (print("DEBUG: End Conversation:", x), "Got it! Feel free to ask me other questions at your convenience. Thank you for using this chatbot. 'Till next time!")[1])
    ),
    # Default branch (if none of the above conditions are met)
    RunnableLambda(lambda x: (print("DEBUG:", x.content), "I'm not sure how to handle that. Can you clarify?")
    )
)

'''
FINAL CHAIN - This will be the main final chain for this classifier
'''
questionClassifier = chain_verifyQuery | qc_Branch

