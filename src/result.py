import os
from dotenv import load_dotenv
from pyprojroot import here
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

# 强制覆盖已存在的环境变量
load_dotenv(override=True)

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

sqldb_directory = here("data/travel.sqlite")
db = SQLDatabase.from_uri(f"sqlite:///{sqldb_directory}")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM aircrafts_data LIMIT 10;")

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

custom_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    You are an SQL expert. Given the input question, generate an SQL query that answers the question.
    The query must be simple and efficient. For example, if the question is about counting rows,
    the query should use SELECT COUNT(*) syntax.
    Question: {question}
    """
)

chain = LLMChain(llm=llm, prompt=custom_prompt)
response = chain.invoke({"question": "How many rows are there in the aircrafts_data table?"})
print(response['text'])

db.run(response)