import openai
import os
from dotenv import load_dotenv
load_dotenv()

# 获取 OpenAI API 密钥并设置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper

# Initialize tools
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for answering questions about current events or the current state of the world"
    ),
]

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize OpenAI language model
llm = OpenAI(temperature=0)

# Initialize the agent
agent_chain = initialize_agent(
    tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory
)

# Run conversations
responses = [
    agent_chain.run(input="hi, i am ZJD"),
    # agent_chain.run(input="what's my name?"),
    # agent_chain.run(input="what are some good dinners to make this week, if i like thai food?"),
    agent_chain.run(input="请告诉我2022世界杯冠军是谁？"),
    # agent_chain.run(input="what's the current temperature in pomfret?")
]

# Print responses
for response in responses:
    print(response)
