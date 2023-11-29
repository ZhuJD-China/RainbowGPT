import openai
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
print(OPENAI_API_KEY)

# 设置代理（替换为你的代理地址和端口）
proxy_url = 'http://localhost:7890'
os.environ['http_proxy'] = proxy_url
os.environ['https_proxy'] = proxy_url


from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()

tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
)

print(tool.run("只告诉我现在以太坊的价格?"))
print(tool.run("现在的时间是什么时候?"))
