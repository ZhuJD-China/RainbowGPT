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

from langchain.agents import XMLAgent, tool, AgentExecutor
from langchain.chat_models import ChatAnthropic
from langchain.chains import LLMChain

# 创建ChatAnthropic实例，使用Claude-2模型
model = ChatAnthropic(model="claude-2")


# 定义一个用于查询天气的工具函数
@tool
def search(query: str) -> str:
    """Search things about current events."""
    return "32 degrees"


# 将工具函数添加到工具列表中
tool_list = [search]

# 创建LLMChain实例，使用XMLAgent进行代理
chain = LLMChain(
    llm=model,
    prompt=XMLAgent.get_default_prompt(),
    output_parser=XMLAgent.get_default_output_parser()
)

# 创建XMLAgent实例，使用工具列表和LLMChain
agent = XMLAgent(tools=tool_list, llm_chain=chain)

# 创建AgentExecutor实例，用于执行代理的操作
agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)

# 运行代理以回答问题
response = agent_executor.run("What's the weather in New York?")
print(response)
