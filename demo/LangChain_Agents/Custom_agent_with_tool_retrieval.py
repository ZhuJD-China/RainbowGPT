import re
import time

import chromadb
import openai
import os
from dotenv import load_dotenv
from langchain import SerpAPIWrapper, GoogleSerperAPIWrapper, FAISS, OpenAI, LLMChain
from langchain.agents import AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import StringPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document, AgentAction, AgentFinish
from typing import List, Union

# 加载环境变量中的 OpenAI API 密钥
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# 打印 API 密钥
print(OPENAI_API_KEY)

# 创建 ChatOpenAI 实例作为底层语言模型
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
embeddings = OpenAIEmbeddings()
GoogleSerper_search = GoogleSerperAPIWrapper()

"""
print("==========doc data vector search=======")
load_choice = input("是否加载之前保存的索引？（输入'yes'或'no'）：")

new_db = None

if load_choice.lower() == 'yes':
    new_db = FAISS.load_local(".faiss_index", embeddings, index_name=f"HTD-ChatGPT")
    # query = input("请输入查询文本：")
    # docs = new_db.similarity_search(query)
    # print(docs[0])
else:
    doc_data_path = input("请输入目标目录路径（按回车使用默认值 ./）：") or "./"
    loader = DirectoryLoader(doc_data_path, show_progress=True, use_multithreading=True, silent_errors=True)
    documents = loader.load()
    print(documents)
    print("documents len= ", len(documents))
    input_chunk_size = input("请输入切分token长度（按回车使用默认值 1536）：") or "1536"
    input_chunk_overlap = input("请输入overlap token长度（按回车使用默认值 0）：") or "0"

    if input_chunk_size.isdigit() and input_chunk_overlap.isdigit():
        embeddings.chunk_size = int(input_chunk_size)
        embeddings.show_progress_bar = True
        embeddings.request_timeout = 20
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=int(input_chunk_size),
                                              chunk_overlap=int(input_chunk_overlap))
        docs = text_splitter.split_documents(documents)
        print(docs)
        print("after split documents len= ", len(docs))
        db = FAISS.from_documents(docs, embeddings)

        query = input("请输入查询文本：")
        docs = db.similarity_search(query)
        print("==============================")
        print(docs[0])

        db.save_local(".faiss_index", index_name=f"HTD-ChatGPT")
    else:
        print("输入的切分长度和overlap长度必须是数字。")
"""

# Define which tools the agent can use to answer user queries
search = SerpAPIWrapper()

search_tool = Tool(
    name="Search",
    func=GoogleSerper_search.run,
    description="useful for when you need to answer questions about current events",
)


def fake_func(inp: str) -> str:
    return "foo"


fake_tools = [
    Tool(
        name=f"foo-{i}",
        func=fake_func,
        description=f"a silly function that you can use to get more information about the number {i}",
    )
    for i in range(99)
]
ALL_TOOLS = [search_tool] + fake_tools

docs = [
    Document(page_content=t.description, metadata={"index": i})
    for i, t in enumerate(ALL_TOOLS)
]

vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

retriever = vector_store.as_retriever()


def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    return [ALL_TOOLS[d.metadata["index"]] for d in docs]


# print(get_tools("有多少章节?"))
# print(get_tools("有多少引用内容?"))
# print(get_tools("whats the weather?"))
# print(get_tools("whats the number 13?"))

# Set up the base template
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""

from typing import Callable


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"],
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


output_parser = CustomOutputParser()

llm = OpenAI(temperature=0)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

tools = get_tools("whats the weather?")
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
    verbose=True,
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
agent_executor.run("最新的浙江省绍兴市的GDP是多少？全国排名是多少？占全国的GDP比例是多少？")
