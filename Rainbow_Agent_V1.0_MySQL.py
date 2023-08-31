import openai
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# 从环境变量加载 OpenAI API 密钥
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from loguru import logger
from langchain.callbacks import FileCallbackHandler


# 提示用户输入数据库名，如果没有输入，则使用默认值
default_db_name = "movies_data"
user_db_name = input(f"请输入数据库名（默认：{default_db_name}）：").strip()
db_name = user_db_name if user_db_name else default_db_name

# 创建数据库连接
db = SQLDatabase.from_uri(f"mysql+pymysql://root:*******@localhost/{db_name}")
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# 创建代理执行器
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
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
        response = agent_executor.run(user_input_text)
        print(response)
    except Exception as e:
        print("发生错误:", e)
