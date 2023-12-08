import os
import dashscope
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
dashscope.api_key = DASHSCOPE_API_KEY
# 打印 API 密钥
print(dashscope.api_key)

from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = ChatOpenAI(
    model_name="Qwen-7B-Chat",
    openai_api_base="http://172.16.0.160:8000/v1",
    openai_api_key="",
    streaming=False,
)
tools = load_tools(
    ["arxiv"],
)
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
# TODO: The performance is okay with Chinese prompts, but not so good when it comes to English.
agent_chain.run("查一下论文 1605.08386 的信息")
