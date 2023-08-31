import openai
import os
from dotenv import load_dotenv

# 加载环境变量文件
load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# 创建一个 OpenAI 语言模型实例
llm = OpenAI(temperature=0)

# 创建一个用于搜索的 SerpAPIWrapper 实例
search = SerpAPIWrapper()

# 创建代理所需的工具，用于搜索问题并获取中间答案
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

# 使用工具、语言模型和指定的代理类型来初始化代理
self_ask_with_search = initialize_agent(
    tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
)

# 要查询的问题
question = "中国总共有几个不同的领导人？"

# 运行代理来回答问题
answer = self_ask_with_search.run(question)

# 输出代理的回答
print(answer)
