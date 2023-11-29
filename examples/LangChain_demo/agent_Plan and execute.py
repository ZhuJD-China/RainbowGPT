"""
ReAct（Recursive Actor-Critic）和Plan and Execute（计划和执行）是两种不同的方法，用于实现目标驱动的任务分解和执行。它们都旨在让代理智能地完成复杂任务，但在实现方式上有一些区别。

ReAct（Recursive Actor-Critic）：

ReAct 是一种深度强化学习方法，用于将大任务逐步分解为更小的子任务，并通过递归地调用自身来实现任务分解。
它基于递归的思想，代理在每个阶段都做出决策来选择下一个子任务，并在子任务完成后，递归调用自身来继续分解子任务，直到达到最终目标。
ReAct 使用强化学习算法，如 Actor-Critic，来训练代理执行任务分解和子任务执行，使得代理可以自主地学习如何选择子任务和执行动作。
Plan and Execute（计划和执行）：

Plan and Execute 是一种通过明确的规划和执行过程来实现任务分解的方法，代理首先通过规划确定任务的子任务和执行顺序，然后按照规划执行子任务。
它可以使用符号规划方法、启发式搜索等来生成任务的执行计划，然后使用工具和执行器来执行计划中的子任务。
Plan and Execute 不一定涉及到强化学习，更多地依赖于规划和工具的结合来完成任务。

"""
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

# 导入所需的模块和类
from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain

# 初始化工具和语言模型
search = SerpAPIWrapper()
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
]

# 加载计划者和执行者，创建代理
model = ChatOpenAI(temperature=0)

planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# 用户输入
user_input = "2022世界杯冠军是谁？梅西在决赛中进了几个球？决赛的两只队伍最终的比分是多少？总比分的和加起来乘以0.9的平方是多少？"

# 运行代理与模型的交互，完成规划和执行
result = agent.run(user_input)
print(result)
