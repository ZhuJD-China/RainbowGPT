import chromadb
import os
from dotenv import load_dotenv
import gradio as gr
# 导入 langchain 模块的相关内容
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
# Rainbow_utils
from Rainbow_utils.get_gradio_theme import Seafoam
from Rainbow_utils.get_tokens_cal_filter import filter_chinese_english_punctuation, num_tokens_from_string, \
    truncate_string_to_max_tokens
from Rainbow_utils import get_google_result
from Rainbow_utils import get_prompt_templates
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from loguru import logger
from langchain.callbacks import FileCallbackHandler

load_dotenv()

seafoam = Seafoam()

script_name = os.path.basename(__file__)
logfile = script_name + ".log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)

# local private llm name
local_private_llm_name_global = None
# private llm apis
local_private_llm_api_global = None
# private llm api key
local_private_llm_key_global = None
# http proxy
proxy_url_global = None
# 创建 ChatOpenAI 实例作为底层语言模型
llm = None
llm_name_global = None
embeddings = None
Embedding_Model_select_global = 0
temperature_num_global = 0
# 本地知识库嵌入token max
local_data_embedding_token_max_global = None
human_input_global = None

local_search_template = get_prompt_templates.local_search_template
google_search_template = get_prompt_templates.google_search_template

# 全局工具列表创建
tools = []

# memory
agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)


def echo(message, history, llm_options_checkbox_group,
         Embedding_Model_select, local_data_embedding_token_max, local_private_llm_api,
         local_private_llm_key,
         local_private_llm_name, input_datatable_name):
    global docsearch_db
    global llm
    global tools
    global RainbowGPT
    global list_collections_name
    global embeddings
    global Embedding_Model_select_global
    global temperature_num_global
    global llm_name_global
    global input_chunk_size_global
    global local_data_embedding_token_max_global
    global human_input_global
    global collection_name_select_global
    global proxy_url_global
    global local_private_llm_api_global
    global local_private_llm_key_global
    global local_private_llm_name_global
    print_speed_step = 10
    temperature_num_global = 0

    local_private_llm_name_global = str(local_private_llm_name)
    local_private_llm_api_global = str(local_private_llm_api)
    local_private_llm_key_global = str(local_private_llm_key)
    human_input_global = message
    local_data_embedding_token_max_global = int(local_data_embedding_token_max)

    llm_name_global = str(llm_options_checkbox_group)

    if Embedding_Model_select == "Openai Embedding" or Embedding_Model_select == "" or Embedding_Model_select == None:
        embeddings = OpenAIEmbeddings()
        embeddings.show_progress_bar = True
        embeddings.request_timeout = 20
        Embedding_Model_select_global = 0
    elif Embedding_Model_select == "HuggingFace Embedding":
        embeddings = HuggingFaceEmbeddings(cache_folder="../models")
        Embedding_Model_select_global = 1

    if llm_name_global == "Private-LLM-Model":
        llm = ChatOpenAI(
            model_name=local_private_llm_name_global,
            openai_api_base=local_private_llm_api_global,
            openai_api_key=local_private_llm_key_global,
            streaming=False,
        )
    else:
        llm = ChatOpenAI(temperature=temperature_num_global,
                         openai_api_key=os.getenv('OPENAI_API_KEY'),
                         model=llm_name_global)

    if message == "":
        response = "哎呀！好像有点小尴尬，您似乎忘记提出问题了。别着急，随时输入您的问题，我将尽力为您提供帮助！"
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]
        return

    db_name = input_datatable_name
    # 创建数据库连接
    db = SQLDatabase.from_uri(f"mysql+pymysql://root:123456@localhost/{db_name}")
    # db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

    # 创建代理执行器
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    try:
        response = agent_executor.run(message)
    except Exception as e:
        response = f"发生错误：{str(e)}"
    for i in range(0, len(response), int(print_speed_step)):
        yield response[: i + int(print_speed_step)]
    # return response


with gr.Blocks(theme=seafoam) as RainbowGPT:
    with gr.Row():
        with gr.Column(scale=3):
            # 左侧列: 所有控件
            with gr.Row():
                with gr.Group():
                    gr.Markdown("### Language Model Selection")
                    llm_options = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo-16k",
                                   "gpt-3.5-turbo", "Private-LLM-Model"]
                    llm_options_checkbox_group = gr.Dropdown(llm_options, label="LLM Model Select Options",
                                                             value=llm_options[0])
                    local_private_llm_name = gr.Textbox(value="Qwen-7B-Chat", label="Private llm name")

                with gr.Group():
                    gr.Markdown("### Private LLM Settings")
                    local_private_llm_api = gr.Textbox(value="http://172.16.0.160:8000/v1",
                                                       label="Private llm openai-api base")
                    local_private_llm_key = gr.Textbox(value="EMPTY", label="Private llm openai-api key")

            with gr.Row():
                with gr.Group():
                    input_datatable_name = gr.Textbox(value="movies_data", label="datatable name")

                    gr.Markdown("### Embedding Data Settings")
                    Embedding_Model_select = gr.Radio(["Openai Embedding", "HuggingFace Embedding"],
                                                      label="Embedding Model Select Options", value="Openai Embedding")
                    local_data_embedding_token_max = gr.Slider(1024, 12288, step=2, label="Embeddings Data Max Tokens",
                                                               value=2048)

        with gr.Column(scale=5):
            # 右侧列: Chat Interface
            gr.ChatInterface(
                echo, additional_inputs=[llm_options_checkbox_group,
                                         Embedding_Model_select, local_data_embedding_token_max, local_private_llm_api,
                                         local_private_llm_key,
                                         local_private_llm_name, input_datatable_name],
                title="""
    <h1 style='text-align: center; margin-bottom: 1rem; font-family: "Courier New", monospace;
               background: linear-gradient(135deg, #9400D3, #4B0082, #0000FF, #008000, #FFFF00, #FF7F00, #FF0000);
               -webkit-background-clip: text;
               color: transparent;'>
        RainbowSQL-Agent
    </h1>
    """,
                description="<div style='font-size: 14px; ...'>How to reach me: <a href='mailto:zhujiadongvip@163.com'>zhujiadongvip@163.com</a></div>",
                # css=".gradio-container {background-color: #f0f0f0;}"  # Add your desired background color here
            )

RainbowGPT.queue().launch()
# RainbowGPT.queue().launch(share=True)
