import openai
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
print(OPENAI_API_KEY)

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import PyPDFLoader

#
# Load the document, split it into chunks, embed each chunk and load it into the vector store.
# raw_documents = TextLoader('../../../state_of_the_union.txt').load()
raw_documents = UnstructuredPDFLoader("D:\AIGC\langchain\data\\bitcoin\Bitcoin.pdf").load()
print(raw_documents)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())
query = "What is the proof-of-work?"

docs = db.similarity_search(query)
print(docs[0].page_content)

embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)

print("--------------------------------------------------------------")
print("--------------------------------------------------------------")
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader

loader = UnstructuredPDFLoader("D:\AIGC\langchain\data\\bitcoin\Bitcoin.pdf")
from langchain.indexes import VectorstoreIndexCreator

# index = VectorstoreIndexCreator().from_loaders([loader])
index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
).from_loaders([loader])
query = "请先告诉我中本聪是谁？？再告诉我Satoshi Nakamoto是谁？？他们俩是同一个人吗？"
print(index_creator.query(query))
