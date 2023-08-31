import openai
import os
from dotenv import load_dotenv

load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

import os
import dotenv
import pydantic
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.schema import AgentFinish
from langchain.agents.tools import Tool
from langchain import LLMMathChain
from langchain.chat_models import ChatOpenAI

# Uncomment if you have a .env in root of repo contains OPENAI_API_KEY
# dotenv.load_dotenv("../../../../../.env")

# 创建ChatOpenAI实例，使用gpt-4模型
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# 定义工具，提供第n个素数和LLMMathChain作为计算器
primes = {998: 7901, 999: 7907, 1000: 7919}


class CalculatorInput(pydantic.BaseModel):
    question: str = pydantic.Field()


class PrimeInput(pydantic.BaseModel):
    n: int = pydantic.Field()


def is_prime(n: int) -> bool:
    if n <= 1 or (n % 2 == 0 and n > 2):
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def get_prime(n: int, primes: dict = primes) -> str:
    return str(primes.get(int(n)))


async def aget_prime(n: int, primes: dict = primes) -> str:
    return str(primes.get(int(n)))


tools = [
    Tool(
        name="GetPrime",
        func=get_prime,
        description="A tool that returns the `n`th prime number",
        args_schema=PrimeInput,
        coroutine=aget_prime,
    ),
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="Useful for when you need to compute mathematical expressions",
        args_schema=CalculatorInput,
        coroutine=llm_math_chain.arun,
    ),
]

# 构建代理
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# 迭代运行代理并检查中间步骤
question = "What is the product of the 998th, 999th and 1000th prime numbers?"

for step in agent.iter(question):
    if output := step.get("intermediate_step"):
        action, value = output[0]
        if action.tool == "GetPrime":
            print(f"Checking whether {value} is prime...")
            assert is_prime(int(value))
        # Ask user if they want to continue
        _continue = input("Should the agent continue (Y/n)?:\n")
        if _continue != "Y":
            break
