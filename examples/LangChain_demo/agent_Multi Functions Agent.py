# 导入所需模块和类
import openai
import os
from dotenv import load_dotenv
load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

from langchain import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI

# 初始化 OpenAI 语言模型和 SerpAPI 包装器
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
search = SerpAPIWrapper()

# 定义工具列表，包括 "Search" 工具
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful when you need to answer questions about current events. You should ask targeted questions.",
    ),
]

# 初始化代理，并配置为 AgentType.OPENAI_MULTI_FUNCTIONS
mrkl = initialize_agent(
    tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True
)

# 运行代理与模型的交互，查询天气情况
result = mrkl.run("What is the weather in 浙江省杭州市 today, yesterday, and the day before?What is the date today")
print(result)

# 配置代理的最大迭代次数和提前停止方法
mrkl_configured = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    max_iterations=2,
    early_stopping_method="generate",
)

# 运行代理与模型的交互，查询天气情况，带有配置的最大迭代次数和提前停止方法
result_configured = mrkl_configured.run("What is the weather in 浙江省杭州市 today, yesterday, and the day before?What is the date today")
print(result_configured)
