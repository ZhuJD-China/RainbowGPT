import openai
import os
from dotenv import load_dotenv
load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

from langchain import (
    LLMMathChain,
    OpenAI,
    SerpAPIWrapper,
    # SQLDatabase,
    # SQLDatabaseChain,
)
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

# 创建ChatOpenAI实例，使用gpt-3.5-turbo-0613模型
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

# 创建SerpAPIWrapper实例
search = SerpAPIWrapper()

# 创建LLMMathChain实例
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# # 创建SQLDatabase实例
# db = SQLDatabase.from_uri("sqlite:///../../../../../notebooks/Chinook.db")
#
# # 创建SQLDatabaseChain实例
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# 创建工具列表
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

# 创建MessagesPlaceholder实例来作为内存的占位符
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

# 初始化OpenAI Functions代理
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

# 运行代理以进行对话
response = agent.run("hi")
print(response)

response = agent.run("my name is bob")
print(response)

response = agent.run("whats my name")
print(response)
