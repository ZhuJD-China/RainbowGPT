import openai
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
print(OPENAI_API_KEY)

from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator

# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# index = VectorstoreIndexCreator().from_loaders([loader])
# answer = index.query("What is Task Decomposition?")
# print(answer)


# from langchain.document_loaders import DirectoryLoader
# from langchain.document_loaders import TextLoader
# text_loader_kwargs={'autodetect_encoding': True}
# loader = DirectoryLoader('../', glob="**/*.mdx", show_progress=True, use_multithreading=True,loader_cls=TextLoader, silent_errors=True)
# docs = loader.load()
# print(docs)


from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import MathpixPDFLoader
loader = PyPDFLoader("D:\AIGC\langchain\data\Bitcoin.pdf")
# loader = UnstructuredPDFLoader("D:\AIGC\langchain\data\Bitcoin.pdf")
# loader = MathpixPDFLoader("D:\AIGC\langchain\data\Bitcoin.pdf")
pages = loader.load_and_split()
print(pages[0])

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
docs = faiss_index.similarity_search("什么是A Peer-to-Peer Electronic系统?", k=2)
for doc in docs:
    # print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
    print(doc)
