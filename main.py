from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableBranch
from langchain_openai import ChatOpenAI

from bookstop_modules import questionclassifier

query = {"query": input('>>: ')}
result = questionclassifier.questionClassifier.invoke(query)
print(result)