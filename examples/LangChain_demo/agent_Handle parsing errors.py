import openai
import os
from dotenv import load_dotenv

load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
# 打印 API 密钥
print(OPENAI_API_KEY)

# 导入所需模块和类
from langchain import (
    OpenAI,
    LLMMathChain,
    SerpAPIWrapper,
)
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents.types import AGENT_TO_CLASS

# 创建搜索工具
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
]

# 初始化代理并处理解析错误
mrkl = initialize_agent(
    tools,
    ChatOpenAI(temperature=0),
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# 运行代理并提供输入
mrkl.run("Who is Leo DiCaprio's girlfriend? No need to add Action")
