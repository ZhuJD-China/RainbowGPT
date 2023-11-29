import requests
from bs4 import BeautifulSoup

url = "https://python.langchain.com/docs/modules/model_io/prompts/"

# 发送GET请求获取网页内容
response = requests.get(url)
content = response.text

# 使用BeautifulSoup解析HTML内容
soup = BeautifulSoup(content, "html.parser")

# print(content)


# 打印网页标题
title = soup.title.text
print("网页标题:", title)

# 打印段落文本内容
paragraphs = soup.find_all("p")
for paragraph in paragraphs:
    print(paragraph.text)



code_txt = soup.find_all("code")
print(code_txt)