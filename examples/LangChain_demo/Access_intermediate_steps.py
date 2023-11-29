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

# 初始化LLM模型和工具
llm = OpenAI(temperature=0, model_name="text-davinci-002")
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 初始化代理并返回中间步骤
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
)

# 运行代理并提供输入
response = agent(
    {
        "input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
    }
)

# 打印中间步骤
print(response["intermediate_steps"])

def serialize_agent_action(agent_action):
    return {
        "tool": agent_action.tool,
        "tool_input": agent_action.tool_input,
        "log": agent_action.log
    }

# 将中间步骤进行序列化处理，以便于JSON输出
serialized_intermediate_steps = [
    (serialize_agent_action(action), observation)
    for action, observation in response["intermediate_steps"]
]

# 打印中间步骤的JSON格式
import json
print(json.dumps(serialized_intermediate_steps, indent=2))
