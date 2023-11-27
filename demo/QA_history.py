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

# 初始化对话历史
dialogue_history = []

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

    # 将用户输入添加到对话历史中
    dialogue_history.append("用户：" + user_input_text)

    # 构建完整的对话历史文本
    conversation = "\n".join(dialogue_history)

    # print('-----------------------------------------')
    # print(conversation)

    print("****Thingking******")
    # 调用OpenAI API进行问答
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": conversation},
        ],
        temperature=1,
    )

    # 解析API响应并获取回答
    answer = response.choices[0].message['content']

    # 将回答添加到对话历史中
    dialogue_history.append("助手：" + answer)

    # 打印回答
    print(answer)
