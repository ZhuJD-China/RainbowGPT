import openai
import os
from dotenv import load_dotenv

load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
SERPER_API_KEY= os.getenv('SERPER_API_KEY')
# 打印 API 密钥
print(OPENAI_API_KEY)
print(SERPER_API_KEY)

import asyncio
import time
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.llms import OpenAI


# Initialize OpenAI LLM
llm = OpenAI(temperature=0)

# Load tools (GoogleSerperAPIWrapper, LLMMathChain, etc.)
tools = load_tools(["google-serper", "llm-math"], llm=llm)

# Initialize the agent
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# List of questions to ask the agent
questions = [
    "Who won the US Open men's final in 2023? What is his age raised to the 0.334 power?",
    "中国人口2023统计数量？",
    "世界500强企业中国上榜几家？"

]

# Serial Execution
s = time.perf_counter()
for q in questions:
    agent.run(q)
elapsed = time.perf_counter() - s
print(f"Serial executed in {elapsed:0.2f} seconds.")


# Concurrent Execution
async def concurrent_execution():
    s = time.perf_counter()
    tasks = [agent.arun(q) for q in questions]
    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - s
    print(f"Concurrent executed in {elapsed:0.2f} seconds.")


# Run concurrent execution using asyncio event loop
asyncio.run(concurrent_execution())
