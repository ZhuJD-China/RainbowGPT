"""

MRKL代理是Langchain平台上的一种代理类型，用于处理自然语言查询并与外部数据源进行交互以获取答案。MRKL代理具有灵活的架构，可以根据不同的任务和需求进行定制。它包含三个关键部分：

工具（Tools）：这是代理可以使用的外部数据源或服务，比如搜索引擎、API接口等。代理可以使用这些工具来获取信息或执行操作。

LLMChain：LLM（Language Model）Chain是一个语言模型链，它将用户的查询和之前的交互输入传递给底层的语言模型。LLMChain产生的文本会被解析，以确定代理应该采取的操作。

代理类（Agent Class）：代理类负责解析LLMChain生成的文本，并根据解析结果决定采取的操作。这包括了使用工具来获取答案、存储交互历史等。

总之，MRKL代理充当了用户与外部数据源之间的桥梁，它能够将用户的自然语言查询转化为实际的数据请求或操作，并返回相应的结果。
代理的灵活性和可定制性使其适用于不同领域和应用场景中，可以根据任务的需求定制特定的代理行为和交互流程。

"""
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

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain

# Create a SerpAPIWrapper tool for searching
search = SerpAPIWrapper()

# Define the available tools for the agent
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]

"""
前缀（Prefix）：前缀是添加在提示文本开头的一段文字。它通常用于引入任务的背景信息、设置问题的上下文或提供指令给语言模型。在代理的提示中，前缀可以包含诸如可用工具列表、任务背景等信息。

后缀（Suffix）：后缀是添加在提示文本结尾的一段文字。它通常用于结束对话、指示模型何时停止生成文本，或提供特定的最终指令。在代理的提示中，后缀通常包含用户的输入、代理的工作记录、最终的回答等。

在创建代理时，前缀和后缀的内容是可以自定义的，这意味着您可以根据任务的要求定制适当的前缀和后缀文本。
这样，您可以在提示中引导语言模型以特定的方式生成文本，确保代理能够根据预期执行任务并生成相应的回答。
"""
# Define the prefix and suffix for the prompt
prefix = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:"""
suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

Question: {input}
{agent_scratchpad}"""

# Create a prompt using the ZeroShotAgent helper method
prompt = ZeroShotAgent.create_prompt(
    tools, prefix=prefix, suffix=suffix, input_variables=["input", "agent_scratchpad"]
)

# Create an LLMChain with OpenAI LLM and the custom prompt
llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)

# Get the names of allowed tools
tool_names = [tool.name for tool in tools]

# Create the ZeroShotAgent using the LLMChain and allowed tools
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

# Create the agent executor with the custom agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

# Run the agent to answer a question
response = agent_executor.run("How many people live in china as of 2023?")
print(response)
