"""
Multi-Hop vectorstore reasoning（多跳向量存储推理）：
这种方法涉及使用代理（Agent）结合向量存储（Vectorstore）来进行多跳问题的推理。多跳推理通常指的是需要通过多个中间步骤来找到答案的问题。
在这种方法中，代理作为一个更复杂的智能体，使用向量存储作为工具来帮助解决问题。
代理可能会进行多个步骤的查询和推理，以便从向量存储中获取信息，然后逐步构建出答案。
这个方法适用于需要进行更多的问题求解步骤，涉及多个中间环节的情况。

Use the Agent solely as a router（仅将代理用作路由器）：
这种方法将代理简化为一个路由器，用于将用户的查询传递给不同的工具，然后直接返回工具的结果。
在这种情况下，代理不会进行复杂的问题求解或推理，只是将工具的结果返回给用户。
这种方法适用于简单的查询场景，其中每个工具能够直接提供所需的信息，而代理只是起到一个中介传递的作用，不对结果进行处理。

"""
import time

import openai
import os
from dotenv import load_dotenv

# 加载环境变量中的 OpenAI API 密钥
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from pathlib import Path
from langchain.document_loaders import TextLoader, WebBaseLoader, DirectoryLoader
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI

# 创建 ChatOpenAI 实例作为底层语言模型
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

# 设置向量存储相关配置
print("==========docsearch=======")
# doc_path = "D:\\AIGC\\langchain\\data\\bitcoin_news.txt"
# loader = TextLoader(doc_path, encoding='utf-8')
# documents = loader.load()
# print(documents)

# Specify the folder path where your documents are located
folder_path = "D:\AIGC\langchain\data\\bitcoin"
# Create a DirectoryLoader instance with the folder path
loader = DirectoryLoader(folder_path, show_progress=True, use_multithreading=True,
                         silent_errors=True)
# Load documents from the specified folder
documents = loader.load()
# Now you have loaded documents from the folder
time.sleep(3)
print(documents)

text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=10)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
# embeddings.chunk_size = 1500
# embeddings.request_timeout = 61
# embeddings.show_progress_bar = True

len_total = 0
for text in texts:
    len_total = len_total + len(text.page_content)
docsearch = Chroma.from_documents(texts, embeddings, collection_name="local_news")
docsearch_ret = docsearch.as_retriever()
state_of_union = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)

# # 创建 WebBaseLoader 实例并加载文档
# print("==========WebBaseLoader=======")
# loader = WebBaseLoader("https://www.8btc.com/flash")
# docs = loader.load()
# print(docs)
# ruff_texts = text_splitter.split_documents(docs)
# ruff_db = Chroma.from_documents(ruff_texts, embeddings, collection_name="online_news")
# ruff = RetrievalQA.from_chain_type(
#     llm=llm, chain_type="stuff", retriever=ruff_db.as_retriever()
# )

# Create a SerpAPIWrapper tool for searching
search = SerpAPIWrapper(search_engine="google")

# 创建工具列表
tools = [
    Tool(
        name="local_database System",
        func=state_of_union.run,
        description="answer first! search local database",
    ),
    Tool(
        name="Google Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),

    # Tool(
    #     name="online_news System",
    #     func=ruff.run,
    #     description="useful for when you need to answer questions. Input should be a fully formed question.",
    # )
]

# 初始化代理
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

while True:
    user_input = []
    print("请输入您的问题（纯文本格式），换行输入 n 以结束：")
    while True:
        line = input()
        if line != "n":
            user_input.append(line)
        else:
            break
    user_input_text = "\n".join(user_input)

    print("*******************************Thingking****************************************")
    response = agent.run(user_input_text)
    print(response)
