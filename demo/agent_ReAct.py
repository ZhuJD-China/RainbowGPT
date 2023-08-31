import openai
import os
from dotenv import load_dotenv

load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

# 导入所需模块和类
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

# 初始化一个 OpenAI 语言模型
llm = OpenAI(temperature=0)

# 加载工具，包括 "serpapi" 和 "llm-math"
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化一个 ReAct 代理，使用上面加载的工具和语言模型
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# 运行代理以执行 ReAct 逻辑
result = agent.run("2022世界杯冠军是谁？梅西在决赛中进了几个球？决赛的两只队伍最终的比分是多少？总比分的和加起来乘以0.9的平方是多少？")

# 打印 ReAct 逻辑的输出
print(result)
