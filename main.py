from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableBranch
from langchain_openai import ChatOpenAI

from bookstop_modules import qc_VerifyQuery

first = RunnableBranch(

)
query = {"query": input('>>: ')}
result = qc_VerifyQuery.chain_verifyQuery.invoke(query)
second = RunnableBranch(

)
print(result)
chain = first | second