import chromadb
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
print(OPENAI_API_KEY)

client = chromadb.PersistentClient(path=".chromadb/")
print(client.list_collections())

# get a collection
collection_name = input("请输入要获取的collection name：")
chroma_collection = client.get_collection(collection_name)
print(chroma_collection.count())

# 创建 ChatOpenAI 实例作为底层语言模型
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
service_context = ServiceContext.from_defaults(llm=llm)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

query_engine = index.as_query_engine(service_context=service_context, verbose=True, streaming=True)

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
    # print(user_input_text)

    # print(user_input_text)
    print("****Thingking******")
    try:
        r = query_engine.query(user_input_text)
        print(r)
    except Exception as e:
        print("出现异常:", str(e))
