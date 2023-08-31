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
from langchain.agents import Tool, AgentExecutor, BaseMultiActionAgent
from langchain import OpenAI, SerpAPIWrapper
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish


# 定义一个随机单词生成函数，用于演示工具
def random_word(query: str) -> str:
    print("\nNow I'm doing this!")
    return "foo"


# 创建一个搜索工具和一个随机单词工具
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="RandomWord",
        func=random_word,
        description="call this to get a random word.",
    ),
]


# 自定义一个多动作代理类
class FakeAgent(BaseMultiActionAgent):

    # 定义输入关键字，这里只有一个关键字 "input"
    @property
    def input_keys(self):
        return ["input"]

    # 实现 plan 方法，根据输入和中间步骤决定执行哪些动作
    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        if len(intermediate_steps) == 0:
            return [
                AgentAction(tool="Search", tool_input=kwargs["input"], log=""),
                AgentAction(tool="RandomWord", tool_input=kwargs["input"], log=""),
            ]
        else:
            return AgentFinish(return_values={"output": "bar"}, log="")

    # 实现异步版的 aplan 方法，与 plan 方法逻辑相同
    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        if len(intermediate_steps) == 0:
            return [
                AgentAction(tool="Search", tool_input=kwargs["input"], log=""),
                AgentAction(tool="RandomWord", tool_input=kwargs["input"], log=""),
            ]
        else:
            return AgentFinish(return_values={"output": "bar"}, log="")


# 创建代理实例
agent = FakeAgent()

# 创建代理执行器，将代理和工具传入
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# 运行代理执行器并提供输入
agent_executor.run("How many people live in china as of 2023?")
