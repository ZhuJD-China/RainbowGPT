import time
import chromadb
import openai
import os
from dotenv import load_dotenv
from langchain_experimental.plan_and_execute import load_chat_planner, load_agent_executor, PlanAndExecute

# 加载环境变量中的 OpenAI API 密钥
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)
from langchain import (
    LLMMathChain,
    OpenAI,
    SerpAPIWrapper,
    SQLDatabase,
)
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
# db = SQLDatabase.from_uri("sqlite:///../../../../../notebooks/Chinook.db")
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    # Tool(
    #     name="FooBar-DB",
    #     func=db_chain.run,
    #     description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context",
    # ),
]
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

agent.run("hi")
agent.run("my name is bob")
agent.run("whats my name")

"""
{'input': 'whats my name', 
'memory': [HumanMessage(content='hi', additional_kwargs={}, example=False), 
AIMessage(content='Hello! How can I assist you today?', additional_kwargs={}, example=False), 
HumanMessage(content='my name is bob', additional_kwargs={}, example=False), 
AIMessage(content='Hello Bob! How can I assist you today?', additional_kwargs={}, example=False)]}
"""
