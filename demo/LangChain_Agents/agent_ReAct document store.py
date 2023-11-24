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

from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer

# 创建一个与 Wikipedia 文档存储进行交互的 DocstoreExplorer 实例
docstore = DocstoreExplorer(Wikipedia())

# 创建代理所需的工具，用于搜索和查找信息
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search",
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup",
    ),
]

# 创建一个 OpenAI 语言模型实例，用于驱动代理的推理
llm = OpenAI(temperature=0, model_name="text-davinci-002")

# 使用工具、语言模型和指定的代理类型来初始化代理
react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)

# 要查询的问题
question = "what is Wikipedia?"

# 运行代理来回答问题
answer = react.run(question)

# 输出代理的回答
print(answer)
