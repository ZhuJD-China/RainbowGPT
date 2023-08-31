# imports
import mwclient  # for downloading example Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
import openai  # for generating embeddings
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
import numpy as np
import docx
import csv
from transformers import GPT2Tokenizer
import codecs


openai_api_key = "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"
openai.api_key = openai_api_key


def read_docx(file_path):
    doc = docx.Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        if 'https://' not in paragraph.text and 'https://' not in paragraph.text:
            text.append(paragraph.text)
    return '\n'.join(text)


file_path = "./data/NBA.docx"
content = read_docx(file_path)

# print(content)

paragraphs = content.split('\n')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

new_section = []

current_section = ""
for par in paragraphs:
    tokens = tokenizer.encode(par, add_special_tokens=False)
    if len(tokens) + len(tokenizer.encode(par, add_special_tokens=False)) > 200:
        current_section += '\n'
        new_section.append(current_section)
        current_section = par
    else:
        current_section += f'\n{par}'

if current_section:
    new_section.append(current_section)


def create_embedding(new_section):
    model_engine = "text-embedding-ada-002"
    openai.api_key = openai_api_key
    response = openai.Embedding.create(
        model=model_engine,
        input=new_section,
    )
    return response['data'][0]['embedding']

print("开始embeddings转换**********************")
embeddings = [create_embedding(index) for index in new_section]


print(new_section)
print("**************************************")
print("开始写入embedding**********************")


df = pd.DataFrame({"text": new_section, "embedding": embeddings})
# save document chunks and embeddings
# 将DataFrame保存到CSV文件中
df.to_csv('./data/embeddings.csv', index=False, encoding='utf-8')

# # 将文本内容编码成 Unicode 字符串
# new_section = [codecs.decode(text, 'unicode_escape') for text in new_section]
# # 将 DataFrame 保存为 CSV 文件
# df = pd.DataFrame({"text": new_section, "embedding": embeddings})
# df.to_csv('embeddings.csv', index=False, encoding='gbk')
