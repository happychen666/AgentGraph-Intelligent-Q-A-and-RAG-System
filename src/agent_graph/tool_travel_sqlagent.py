import os
from typing import List

from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_openai import ChatOpenAI
from agent_graph.load_tools_config import LoadToolsConfig

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_BASE'] = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE= os.environ.get("OPENAI_API_BASE")
TOOLS_CFG = LoadToolsConfig()


class TravelSQLAgentTool:
    """
    A tool for interacting with a travel-related SQL database using an LLM (Language Model) to generate and execute SQL queries.

    This tool enables users to ask travel-related questions, which are transformed into SQL queries by a language model.
    The SQL queries are executed on the provided SQLite database, and the results are processed by the language model to
    generate a final answer for the user.

    Attributes:
        sql_agent_llm (ChatOpenAI): An instance of a ChatOpenAI language model used to generate and process SQL queries.
        system_role (str): A system prompt template that guides the language model in answering user questions based on SQL query results.
        db (SQLDatabase): An instance of the SQL database used to execute queries.
        chain (RunnablePassthrough): A chain of operations that creates SQL queries, executes them, and generates a response.

    Methods:
        __init__: Initializes the TravelSQLAgentTool by setting up the language model, SQL database, and query-answering pipeline.
    """

    def __init__(self, llm: str, sqldb_directory: str, llm_temerature: float) -> None:
        """
        Initializes the TravelSQLAgentTool with the necessary configurations.

        Args:
            llm (str): The name of the language model to be used for generating and interpreting SQL queries.
            sqldb_directory (str): The directory path where the SQLite database is stored.
            llm_temerature (float): The temperature setting for the language model, controlling response randomness.
        """
        self.sql_agent_llm = ChatOpenAI(
            openai_api_base=OPENAI_API_BASE,
            openai_api_key=OPENAI_API_KEY,
            model=llm, 
            temperature=llm_temerature)
        
        self.system_role = """Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n
            Question: {question}\n
            SQL Query: {query}\n
            SQL Result: {result}\n
            Answer:
            """
        self.db = SQLDatabase.from_uri(
            f"sqlite:///{sqldb_directory}",
            sample_rows_in_table_info=0  # 禁用采样行
            )
        print(self.db.get_usable_table_names())
        # 定义自定义提示模板，用于生成 SQL 查询
        custom_prompt = PromptTemplate(
            input_variables=["dialect", "input", "table_info", "top_k"],
            template="""You are a SQL expert using {dialect}.
        Given the following table schema:
        {table_info}
        Generate a syntactically correct SQL query to answer the question: "{input}".
        Limit the results to at most {top_k} rows.
        Return only the SQL query without any additional commentary or Markdown formatting.
        """
        )
        execute_query = QuerySQLDataBaseTool(db=self.db)
        write_query = create_sql_query_chain(
            self.sql_agent_llm, self.db,prompt=custom_prompt)
        answer_prompt = PromptTemplate.from_template(
            self.system_role)
        # 利用 bind 将固定参数绑定到 SQL 查询链中
        bound_chain = write_query.bind(
            dialect=self.db.dialect,
            table_info=self.db.get_table_info(),
            top_k=55
        )
        answer = answer_prompt | self.sql_agent_llm | StrOutputParser()
        # self.chain = (
        #     RunnablePassthrough.assign(query=write_query).assign(
        #         result=itemgetter("query") | execute_query
        #     )
        #     | answer
        # )
        def log_sql(query):
            print("Executing SQL:\n", query)
            return query

        self.chain = (
            RunnablePassthrough.assign(query=bound_chain).assign(
                sql_query=lambda x: log_sql(x["query"]),  # 先打印 SQL 语句
                result=itemgetter("query") | execute_query
            )
            | answer
        )


@tool
def query_travel_sqldb(query: str) -> str:
    """Query the Swiss Airline SQL Database and access all the company's information. Input should be a search query."""
    agent = TravelSQLAgentTool(
        llm=TOOLS_CFG.travel_sqlagent_llm,
        sqldb_directory=TOOLS_CFG.travel_sqldb_directory,
        llm_temerature=TOOLS_CFG.travel_sqlagent_llm_temperature
    )
    response = agent.chain.invoke({"question": query})
    return response

result = query_travel_sqldb('What type of seat is available on the flight?')
print(result)