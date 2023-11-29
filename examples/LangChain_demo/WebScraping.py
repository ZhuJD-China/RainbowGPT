import time
import chromadb
import openai
import os
from dotenv import load_dotenv

# 加载环境变量中的 OpenAI API 密钥
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
# 打印 API 密钥
print(OPENAI_API_KEY)

# from langchain.document_loaders import AsyncChromiumLoader
# from langchain.document_transformers import BeautifulSoupTransformer
#
# # Load HTML
# loader = AsyncChromiumLoader(["https://www.wsj.com"])
# html = loader.load()
#
# # Transform
# bs_transformer = BeautifulSoupTransformer()
# docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["span"])
#
# # Result
# print(docs_transformed[0].page_content[0:500])
#
# from langchain.document_loaders import AsyncHtmlLoader
#
# urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
# loader = AsyncHtmlLoader(urls)
# docs = loader.load()
#
# from langchain.document_loaders import AsyncHtmlLoader
#
# urls = ["https://www.espn.com", "https://lilianweng.github.io/posts/2023-06-23-agent/"]
# loader = AsyncHtmlLoader(urls)
# docs = loader.load()
# print(docs)
#
# from langchain.document_transformers import Html2TextTransformer
#
# html2text = Html2TextTransformer()
# docs_transformed = html2text.transform_documents(docs)
# print(docs_transformed[0].page_content[0:500])
#
# from langchain.chat_models import ChatOpenAI
#
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
#
# from langchain.chains import create_extraction_chain
#
# schema = {
#     "properties": {
#         "news_article_title": {"type": "string"},
#         "news_article_summary": {"type": "string"},
#     },
#     "required": ["news_article_title", "news_article_summary"],
# }
#
#
# def extract(content: str, schema: dict):
#     return create_extraction_chain(schema=schema, llm=llm).run(content)
#
#
# import pprint
# from langchain.text_splitter import RecursiveCharacterTextSplitter
#
#
# def scrape_with_playwright(urls, schema):
#     loader = AsyncChromiumLoader(urls)
#     docs = loader.load()
#     bs_transformer = BeautifulSoupTransformer()
#     docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["span"])
#     print("Extracting content with LLM")
#
#     # Grab the first 1000 tokens of the site
#     splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,
#                                                                     chunk_overlap=0)
#     splits = splitter.split_documents(docs_transformed)
#
#     # Process the first split
#     extracted_content = extract(
#         schema=schema, content=splits[0].page_content
#     )
#     pprint.pprint(extracted_content)
#     return extracted_content
#
#
# urls = ["https://www.wsj.com"]
# extracted_content = scrape_with_playwright(urls, schema=schema)
# print(extracted_content)

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever

# Vectorstore
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai")

# LLM
llm = ChatOpenAI(temperature=0)

# Search
search = GoogleSearchAPIWrapper()

# Initialize
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search)

# Run
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
from langchain.chains import RetrievalQAWithSourcesChain

user_input = "How do LLM Powered Autonomous Agents work?"
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_research_retriever)
result = qa_chain({"question": user_input})
print(result)
