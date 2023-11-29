import openai
import os
from dotenv import load_dotenv
load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

# 导入代理工具装饰器
from langchain.agents import tool

# 定义计算单词长度的工具函数
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

# 创建工具列表，将工具函数添加到其中
tools = [get_word_length]

# 导入 ChatOpenAI 语言模型
from langchain.chat_models import ChatOpenAI

# 初始化 ChatOpenAI 语言模型，设置温度参数
llm = ChatOpenAI(temperature=0)

# 导入系统消息和代理类
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent

# 创建系统消息，用于代理的提示
system_message = SystemMessage(content="You are very powerful assistant, but bad at calculating lengths of words.")

# 在提示中添加内存位置
from langchain.prompts import MessagesPlaceholder
MEMORY_KEY = "chat_history"
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)]
)

# 创建内存对象
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)

# 使用系统消息和提示创建代理
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

# 创建代理执行者
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# 运行代理并与用户交互
user_input = ""
while user_input.lower() != "exit":
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    response = agent_executor.run(user_input)
    print(f"Agent: {response}")