import openai
import os
from dotenv import load_dotenv
load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

# Create an OpenAI chat model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Initialize SerpAPIWrapper for search
search = SerpAPIWrapper()

# Create LLMMathChain for math operations
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# Create SQLDatabaseChain for database queries
# db = SQLDatabase.from_uri("sqlite:/../../../../../notebooks/Chinook.db")
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# Define tools with their functions and descriptions
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for answering questions about current events. You should ask targeted questions."
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for answering math questions."
    ),
    # Tool(
    #     name="FooBar-DB",
    #     func=db_chain.run,
    #     description="useful for answering questions about FooBar. Input should include full context."
    # )
]

# Initialize the OpenAI Functions Agent
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

# Run the agent with a user query
response = agent.run("2022世界杯冠军是谁？梅西在决赛中进了几个球？决赛的两只队伍最终的比分是多少？总比分的和加起来乘以0.9的平方是多少？")
print(response)

