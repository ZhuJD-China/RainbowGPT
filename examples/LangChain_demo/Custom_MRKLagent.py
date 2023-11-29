"""
前缀（Prefix）：前缀是添加在提示文本开头的一段文字。它通常用于引入任务的背景信息、设置问题的上下文或提供指令给语言模型。在代理的提示中，前缀可以包含诸如可用工具列表、任务背景等信息。

后缀（Suffix）：后缀是添加在提示文本结尾的一段文字。它通常用于结束对话、指示模型何时停止生成文本，或提供特定的最终指令。在代理的提示中，后缀通常包含用户的输入、代理的工作记录、最终的回答等。

在创建代理时，前缀和后缀的内容是可以自定义的，这意味着您可以根据任务的要求定制适当的前缀和后缀文本。
这样，您可以在提示中引导语言模型以特定的方式生成文本，确保代理能够根据预期执行任务并生成相应的回答。
"""
import openai
import os
from dotenv import load_dotenv

# 加载环境变量中的 OpenAI API 密钥
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMChain

search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]

prefix = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:"""
suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, prefix=prefix, suffix=suffix, input_variables=["input", "agent_scratchpad"]
)
print(prompt.template)
llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
agent_executor.run("How many people live in canada as of 2023?")

"""
Multiple inputs 多个输入
"""
prefix = """Answer the following questions as best you can. You have access to the following tools:"""
suffix = """When answering, you MUST speak in the following language: {language}.

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "language", "agent_scratchpad"],
)

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)

agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

agent_executor.run(
    input="How many people live in canada as of 2023?", language="italian"
)
