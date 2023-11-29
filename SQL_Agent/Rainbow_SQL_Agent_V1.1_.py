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


few_shots = {'List all artists.': 'SELECT * FROM artists;',
             "Find all albums for the artist 'AC/DC'.": "SELECT * FROM albums WHERE ArtistId = (SELECT ArtistId FROM artists WHERE Name = 'AC/DC');",
             "List all tracks in the 'Rock' genre.": "SELECT * FROM tracks WHERE GenreId = (SELECT GenreId FROM genres WHERE Name = 'Rock');",
             'Find the total duration of all tracks.': 'SELECT SUM(Milliseconds) FROM tracks;',
             'List all customers from Canada.': "SELECT * FROM customers WHERE Country = 'Canada';",
             'How many tracks are there in the album with ID 5?': 'SELECT COUNT(*) FROM tracks WHERE AlbumId = 5;',
             'Find the total number of invoices.': 'SELECT COUNT(*) FROM invoices;',
             'List all tracks that are longer than 5 minutes.': 'SELECT * FROM tracks WHERE Milliseconds > 300000;',
             'Who are the top 5 customers by total purchase?': 'SELECT CustomerId, SUM(Total) AS TotalPurchase FROM invoices GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;',
             'Which albums are from the year 2000?': "SELECT * FROM albums WHERE strftime('%Y', ReleaseDate) = '2000';",
             'How many employees are there': 'SELECT COUNT(*) FROM "employee";',
             '列出 popularity 值最高的十部电影': 'SELECT title, popularity FROM movies ORDER BY popularity DESC LIMIT 10;',
             }
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

embeddings = OpenAIEmbeddings()

few_shot_docs = [Document(page_content=question, metadata={'sql_query': few_shots[question]}) for question in
                 few_shots.keys()]
vector_db = FAISS.from_documents(few_shot_docs, embeddings)
retriever = vector_db.as_retriever()

from langchain.agents.agent_toolkits import create_retriever_tool

tool_description = """
This tool will help you understand similar examples to adapt them to the user question.
Input to this tool should be the user question.
"""

retriever_tool = create_retriever_tool(
    retriever,
    name='sql_get_similar_examples',
    description=tool_description
)
custom_tool_list = [retriever_tool]

from langchain.agents import create_sql_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI

# 提示用户输入数据库名，如果没有输入，则使用默认值
default_db_name = "movies_data"
user_db_name = input(f"请输入数据库名（默认：{default_db_name}）：").strip()
db_name = user_db_name if user_db_name else default_db_name
# 创建数据库连接
db = SQLDatabase.from_uri(f"mysql+pymysql://root:123456@localhost/{db_name}")
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

custom_suffix = """
I should first get the similar examples I know.
If the examples are enough to construct the query, I can build it.
Otherwise, I can then look at the tables in the database to see what I can query.
Then I should query the schema of the most relevant tables
"""

agent = create_sql_agent(llm=llm,
                         toolkit=toolkit,
                         verbose=True,
                         agent_type=AgentType.OPENAI_FUNCTIONS,
                         extra_tools=custom_tool_list,
                         suffix=custom_suffix
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
        response = agent.run(user_input_text)
        print(response)
    except Exception as e:
        print("发生错误:", e)

