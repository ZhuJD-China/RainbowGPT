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
from sklearn.metrics.pairwise import cosine_similarity

openai_api_key = "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"
openai.api_key = openai_api_key


def create_embedding(new_section):
    model_engine = "text-embedding-ada-002"
    openai.api_key = openai_api_key
    response = openai.Embedding.create(
        model=model_engine,
        input=new_section,
    )
    return response['data'][0]['embedding']


def cal_sosine_sim(embeddings1, embeddings2):
    return cosine_similarity([embeddings1], [embeddings2])[0][0]


df = pd.read_csv('./data/embeddings.csv', encoding='utf-8')
print(df)
# 将每一列的字符串转换成浮点数列表
df['embedding'] = df['embedding'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])


def ask_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你作为一个问答助手"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        n=1,
        temperature=1,
    )
    return response.choices[0].message['content'].strip()


while True:
    # 接收中文字符的输入
    q = input('请输入你的问题：').encode('gbk').decode('gbk')
    print(q)

    q_embeddings = create_embedding(q)
    content_embeddings = df['embedding']
    # print(type(content_embeddings))
    # print((content_embeddings))
    similarities = [cal_sosine_sim(q_embeddings, emb) for emb in content_embeddings]
    top_index = np.argmax(similarities)
    top_similarities = similarities[top_index]
    top_text = df.iloc[top_index]["text"]
    prompt = f"使用以下内容回答提出的问题，如果无法回答问题回复'无法找到答案':\n{top_text}\n我的问题是:{q}"
    # print(prompt)
    answer = ask_gpt(prompt)
    print("GPT回答---------------------------------------------------------------------------")
    print(answer)