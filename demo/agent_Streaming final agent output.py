"""
"Streaming final agent output" 是一种在代理生成最终输出时实时流式传输该输出的方法。
这在处理长时间运行的任务或需要及时获取结果的情况下非常有用。以下是使用这种方法的一些主要作用：

实时性： 在代理生成最终输出的同时，可以立即将结果实时传输到用户界面或其他应用程序中，而不需要等待整个代理完成。

节省内存： 对于长时间运行的任务，将结果实时传输可以减少内存占用，因为不需要等到整个任务完成后才生成结果并保存在内存中。

交互性： 在生成过程中流式传输输出，可以与用户进行更多的交互。例如，在代理运行过程中，用户可能想要终止任务或调整参数，这样可以更及时地进行操作。

监控和调试： 实时流式传输输出可以帮助监控代理的进展和性能，并在需要时进行调试。

处理大量数据： 当代理处理大量数据时，通过流式传输输出可以避免等待时间过长，同时也能及时处理并显示部分结果。

总之，"Streaming final agent output" 提供了一种实时获取代理生成结果的方式，使得应用程序和用户能够更加及时地获得结果，提高了交互性和用户体验。

"""
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

print("============================================================")
print("==================创建一个支持流式传输的 LLM，并且添加 FinalStreamingStdOutCallbackHandler 回调==========================================")

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.llms import OpenAI

# 创建一个支持流式传输的 LLM，并且添加 FinalStreamingStdOutCallbackHandler 回调
llm = OpenAI(
    streaming=True, callbacks=[FinalStreamingStdOutCallbackHandler()], temperature=0
)

# 加载工具和初始化代理
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

# 运行代理并流式传输最终输出
final_output = agent.run(
    "It's 2023 now.多少年前习近平当上了中国最高领导人？"
)

print("Final Output:", final_output)

print("============================================================")
print("==================创建一个支持流式传输的 LLM，并指定自定义答案前缀==========================================")


from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.llms import OpenAI

# 创建一个支持流式传输的 LLM，并指定自定义答案前缀
llm = OpenAI(
    streaming=True,
    callbacks=[
        FinalStreamingStdOutCallbackHandler(answer_prefix_tokens=["The", "answer", ":"])
    ],
    temperature=0,
)

# 加载工具和初始化代理
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

# 运行代理并流式传输最终输出
final_output = agent.run(
    "It's 2023 now.多少年前习近平当上了中国最高领导人？"
)

print("Final Output:", final_output)

print("============================================================")
print("=====================创建一个支持流式传输的 LLM，同时传输答案前缀=======================================")


from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.llms import OpenAI

# 创建一个支持流式传输的 LLM，同时传输答案前缀
llm = OpenAI(
    streaming=True,
    callbacks=[
        FinalStreamingStdOutCallbackHandler(
            answer_prefix_tokens=["The", "answer", ":"], stream_prefix=True
        )
    ],
    temperature=0,
)

# 加载工具和初始化代理
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

# 运行代理并流式传输最终输出
final_output = agent.run(
    "It's 2023 now.多少年前习近平当上了中国最高领导人？"
)

print("Final Output:", final_output)
