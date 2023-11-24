import time
import chromadb
import openai
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.plan_and_execute import load_chat_planner, load_agent_executor, PlanAndExecute

# 加载环境变量中的 OpenAI API 密钥
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from pathlib import Path
from langchain.document_loaders import TextLoader, WebBaseLoader, DirectoryLoader, CSVLoader
from langchain.agents import initialize_agent, Tool, load_tools
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from loguru import logger
from langchain.callbacks import FileCallbackHandler, StdOutCallbackHandler
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import get_all_tool_names

logfile = "Rainbow_Agent_output.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)

# 创建 ChatOpenAI 实例作为底层语言模型
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

# embeddings = OpenAIEmbeddings()
# Get embeddings.
embeddings = HuggingFaceEmbeddings()

persist_directory = ".chromadb/"
client = chromadb.PersistentClient(path=persist_directory)
print(client.list_collections())

# Get user input for collection name
collection_name = input("Enter the collection name: ")
try:
    chroma_collection = client.get_collection(collection_name)
    print(chroma_collection.count())
    collection_name_flag = True
except ValueError as e:
    print(e)  # 打印错误消息
    collection_name_flag = False

# Check if the collection exists
if collection_name_flag:
    # Collection exists, load it
    docsearch = Chroma(client=client, embedding_function=embeddings, collection_name=collection_name)
else:
    # 设置向量存储相关配置
    print("==========doc data vector search=======")
    doc_data_path = input("请输入目标目录路径（按回车使用默认值 ./）：") or "./"

    loader = DirectoryLoader(doc_data_path, show_progress=True, use_multithreading=True, silent_errors=True)
    # loader = CSVLoader(doc_data_path, encoding='utf-8')

    documents = loader.load()
    print(documents)
    print("documents len= ", documents.__len__())
    input_chunk_size = input("请输入切分token长度（按回车使用默认值 4000）：") or "4000"
    intput_chunk_overlap = input("请输入overlap token长度（按回车使用默认值 0）：") or "200"
    # embeddings.chunk_size = int(input_chunk_size)
    # embeddings.show_progress_bar = True
    # embeddings.request_timeout = 20

    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=int(input_chunk_size),
                                          chunk_overlap=int(intput_chunk_overlap))
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(input_chunk_size),
    #                                                chunk_overlap=int(intput_chunk_overlap))

    texts = text_splitter.split_documents(documents)
    print(texts)
    print("after split documents len= ", texts.__len__())
    # Collection does not exist, create it
    docsearch = Chroma.from_documents(documents=texts, embedding=embeddings, collection_name=collection_name,
                                      persist_directory=persist_directory)

# 创建MessagesPlaceholder实例来作为内存的占位符
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, \
    ConversationSummaryBufferMemory

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

# 工具集
docsearch_ret = docsearch.as_retriever()
local_database = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)

# 创建工具列表
local_database_tool = Tool(
    # name="Intermediate Answer",
    # name="Lookup",
    name="Search",
    func=local_database.run,
    description="useful for when you need to answer questions about current events",
)

# tools = load_tools(["google-serper", "llm-math", "wikipedia",
#                     "terminal", "python_repl", "arxiv"], llm=llm)
# tools = load_tools(["google-serper", "llm-math",
#                     "terminal", "python_repl", "arxiv"], llm=llm)
tools = []
tools.append(local_database_tool)

# 初始化agent代理
agent_open_functions = initialize_agent(
    tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    max_iterations=10,
    early_stopping_method="generate",
    handle_parsing_errors=True,  # 初始化代理并处理解析错误
    callbacks=[handler],
)
# plan and execut
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)
agent_planner_executor = PlanAndExecute(
    planner=planner, executor=executor, verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
    max_iterations=10,
    early_stopping_method="generate",
    handle_parsing_errors=True,  # 初始化代理并处理解析错误
    callbacks=[handler],
)

# print("=============测试模型记忆能力================")
# response = agent.run("my name is zjd")
# print(response)
# response = agent.run("whats my name")
# print(response)

"""
获取所有工具集名称
from langchain.agents import get_all_tool_names
print(get_all_tool_names())
"""

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

    print("===============Thinking===================")
    try:
        # response = agent_planner_executor.run(user_input_text)
        response = agent_open_functions.run(user_input_text)

        # logger.info(response)
    except Exception as e:
        print("An error occurred:", e)
