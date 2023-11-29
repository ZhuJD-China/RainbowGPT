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

import os
os.environ["LANGCHAIN_TRACING"] = "true"  # If you want to trace the execution of the program, set to "true"

from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent

# 导入PlayWrightBrowserToolkit和相关工具
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.
)

# 仅在jupyter笔记本中需要这个导入，因为它们有自己的事件循环
import nest_asyncio
nest_asyncio.apply()

async def main():
    # 创建异步的PlayWright浏览器实例
    async_browser = create_async_playwright_browser()
    browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = browser_toolkit.get_tools()

    # 创建ChatOpenAI实例
    llm = ChatOpenAI(temperature=0)  # 也可以适用于Anthropic模型

    # 使用工具、语言模型和指定的代理类型来初始化代理
    agent_chain = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # 要查询的问题
    input_text = "Hi I'm Erica."

    # 运行代理来回答问题
    response = await agent_chain.arun(input=input_text)

    # 输出代理的回答
    print(response)

# 异步函数需要通过事件循环来运行
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
