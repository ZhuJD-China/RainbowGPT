import time

import chromadb
import openai
import os
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain, GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from loguru import logger
from langchain.callbacks import FileCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import GoogleSerperRun, Tool
from langchain.vectorstores import Chroma
from transformers import GPT2Tokenizer
import logging

# 加载环境变量中的 OpenAI API 密钥
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)


logfile = "Rainbow_Agent_V1.1_output.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)

# 创建 ChatOpenAI 实例作为底层语言模型
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings()

GoogleSerper_search = GoogleSerperAPIWrapper()

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
    docsearch_db = Chroma(client=client, embedding_function=embeddings, collection_name=collection_name)

    # Filtering_metadata = docsearch_db.get(where={"source": "some_other_source"})
    # print(Filtering_metadata)

else:
    # 设置向量存储相关配置
    print("==========doc data vector search=======")
    doc_data_path = input("请输入目标目录路径（按回车使用默认值 ./）：") or "./"

    loader = DirectoryLoader(doc_data_path, show_progress=True, use_multithreading=True, silent_errors=True)
    # loader = CSVLoader(doc_data_path, encoding='utf-8')

    documents = loader.load()
    print(documents)
    print("documents len= ", documents.__len__())

    input_chunk_size = input("请输入切分token长度（按回车使用默认值 1536）：") or "1536"
    intput_chunk_overlap = input("请输入overlap token长度（按回车使用默认值 0）：") or "0"
    embeddings.chunk_size = int(input_chunk_size)
    embeddings.show_progress_bar = True
    embeddings.request_timeout = 20

    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=int(input_chunk_size),
                                          chunk_overlap=int(intput_chunk_overlap))
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(input_chunk_size),
    #                                                chunk_overlap=int(intput_chunk_overlap))

    texts = text_splitter.split_documents(documents)
    print(texts)
    print("after split documents len= ", texts.__len__())
    # Collection does not exist, create it
    docsearch_db = Chroma.from_documents(documents=texts, embedding=embeddings, collection_name=collection_name,
                                         persist_directory=persist_directory)

# Local_Search Prompt模版
local_search_template = """
你作为一个AI问答助手。
必须通过以下双引号内的知识库内容进行问答:
“{combined_text}”

如果无法回答问题则回复:无法找到答案
我的问题是: {human_input}

"""
local_search_prompt = PromptTemplate(
    input_variables=["combined_text", "human_input"],
    template=local_search_template,
)
# 本地知识库工具
local_chain = LLMChain(
    llm=llm, prompt=local_search_prompt,
    verbose=True,
)

# 使用预训练的gpt2分词器
tokenizers = GPT2Tokenizer.from_pretrained("gpt2")


# 逐字打印
def print_char_by_char(answer):
    for char in answer:
        print(char, end='', flush=True)
        time.sleep(0.01)


# Helper function for printing docs

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
    print("\n====================搜索匹配完成==============\n")


def ask_local_vector_db(question):
    # old docsearch_db
    # docs = docsearch_db.similarity_search(question, k=10)
    # pretty_print_docs(docs)
    # print("**************************************************")

    # new docsearch_db 结合基础检索器+Embedding 压缩+BM25 关检词检索筛选
    chroma_retriever = docsearch_db.as_retriever(search_kwargs={"k": 50})
    # chroma_retriever = docsearch_db.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={'k': 30, 'fetch_k': 50}
    # )

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                           base_retriever=chroma_retriever)
    compressed_docs = compression_retriever.get_relevant_documents(question)
    bm25_Retriever = BM25Retriever.from_documents(compressed_docs)
    bm25_Retriever.k = 30
    docs = bm25_Retriever.get_relevant_documents(question)
    pretty_print_docs(docs)

    cleaned_matches = []
    total_toknes = 0
    # print(docs)
    for context in docs:
        cleaned_context = context.page_content.replace('\n', ' ').strip()

        cleaned_context = f"{cleaned_context}"
        tokens = tokenizers.encode(cleaned_context, add_special_tokens=False)
        if total_toknes + len(tokens) <= (1536 * 8):
            cleaned_matches.append(cleaned_context)
            total_toknes += len(tokens)
        else:
            break

    # 将清理过的匹配项组合合成一个字符串
    combined_text = " ".join(cleaned_matches)

    answer = local_chain.predict(combined_text=combined_text, human_input=question)
    return answer


# 创建工具列表
tools = [
    Tool(
        name="Google_Search",
        func=GoogleSerper_search.run,
        description="""
        当你用本地向量数据库问答后说无法找到答案的之后，你可以使用互联网搜索引擎工具进行信息查询,尝试直接找到问题答案。 
        注意你需要提出非常有针对性准确的问题。
        """,
    ),
    Tool(
        name="Local_Search",
        func=ask_local_vector_db,
        description="""
        你可以首先通过本地向量数据知识库尝试寻找问答案。 
        注意你需要提出非常有针对性准确的问题
        """
    )
]

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

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
        response = agent_open_functions.run(user_input_text)
    except Exception as e:
        print("An error occurred:", e)
