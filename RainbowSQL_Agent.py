import os
from dotenv import load_dotenv
import gradio as gr
# 导入 langchain 模块的相关内容
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from sqlalchemy import create_engine
# Rainbow_utils
from langchain.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from loguru import logger
from langchain.callbacks import FileCallbackHandler


class RainbowSQLAgent:
    def __init__(self):
        self.load_dotenv()
        self.initialize_variables()
        self.create_interface()

    def load_dotenv(self):
        load_dotenv()

    def initialize_variables(self):
        self.logger = logger
        self.script_name = os.path.basename(__file__)
        self.logfile = "./logs/" + self.script_name + ".log"
        logger.add(self.logfile, colorize=True, enqueue=True)
        self.handler = FileCallbackHandler(self.logfile)
        self.local_private_llm_name_global = None
        self.local_private_llm_api_global = None
        self.local_private_llm_key_global = None
        self.proxy_url_global = None
        self.llm = None
        self.llm_name_global = None
        self.temperature_num_global = 0
        self.human_input_global = None
        self.agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }
        self.memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    def get_database_tables(self, host, username, password):
        try:
            # 构建数据库连接字符串
            connection_string = f"mysql+pymysql://{username}:{password}@{host}/"
            # 创建数据库引擎
            engine = create_engine(connection_string)
            # 获取数据库连接
            connection = engine.connect()
            # 查询所有数据库
            result = connection.execute("SHOW DATABASES")
            # 获取查询结果
            databases = [row[0] for row in result]
            # # 打印数据库列表
            # print("Databases:")
            # for db in databases:
            #     print(db)
            # 关闭连接
            connection.close()

            return databases
        except Exception as e:
            print(f"Error: {e}")
            return []

    # 函数：用于更新下拉列表的表格名称
    def update_tables_list(self, host, username, password):
        return gr.Dropdown.update(choices=self.get_database_tables(host, username, password))

    def echo(self, message, history, llm_options_checkbox_group,
             local_private_llm_api,
             local_private_llm_key, local_private_llm_name, input_datatable_name,
             input_database_url, input_database_name, input_database_passwd):
        print_speed_step = 10
        temperature_num_global = 0

        self.local_private_llm_name_global = str(local_private_llm_name)
        self.local_private_llm_api_global = str(local_private_llm_api)
        self.local_private_llm_key_global = str(local_private_llm_key)
        self.human_input_global = message
        self.llm_name_global = str(llm_options_checkbox_group)

        if self.llm_name_global == "Private-LLM-Model":
            llm = ChatOpenAI(
                model_name=self.local_private_llm_name_global,
                openai_api_base=self.local_private_llm_api_global,
                openai_api_key=self.local_private_llm_key_global,
                streaming=False,
            )
        else:
            llm = ChatOpenAI(temperature=temperature_num_global,
                             openai_api_key=os.getenv('OPENAI_API_KEY'),
                             model=self.llm_name_global)

        if message == "":
            response = "哎呀！好像有点小尴尬，您似乎忘记提出问题了。别着急，随时输入您的问题，我将尽力为您提供帮助！"
            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]
            return

        db_name = input_datatable_name
        # 创建数据库连接
        db = SQLDatabase.from_uri(
            f"mysql+pymysql://{input_database_name}:{input_database_passwd}@{input_database_url}/{db_name}")

        # 创建代理执行器
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=db, llm=llm),
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            agent_kwargs=self.agent_kwargs,
            memory=self.memory,
            max_iterations=10,
            early_stopping_method="generate",
            callbacks=[self.handler],
        )

        try:
            response = agent_executor.run(message)
        except Exception as e:
            response = f"发生错误：{str(e)}"
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]
        # return response

    def create_interface(self):
        with gr.Blocks() as self.interface:
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
                            local_private_llm_name = gr.Textbox(value="Qwen-72B-Chat", label="Private llm name")

                        with gr.Group():
                            gr.Markdown("### Private LLM Settings")
                            local_private_llm_api = gr.Textbox(value="http://172.16.0.160:8000/v1",
                                                               label="Private llm openai-api base")
                            local_private_llm_key = gr.Textbox(value="EMPTY", label="Private llm openai-api key")

                    with gr.Row():
                        with gr.Group():
                            gr.Markdown("### DataBase Settings")
                            input_database_url = gr.Textbox(value="localhost", label="MySql Database url")
                            with gr.Row():
                                input_database_name = gr.Textbox(value="root", label="database user name")
                                input_database_passwd = gr.Textbox(value="", label="database user passwd",
                                                                   type="password")
                            input_datatable_name = gr.Dropdown(["..."], label="Database Select Name", value="...")
                            update_button = gr.Button("Update Tables List")
                            update_button.click(fn=self.update_tables_list,
                                                inputs=[input_database_url, input_database_name,
                                                        input_database_passwd],
                                                outputs=input_datatable_name)
                with gr.Column(scale=5):
                    # 右侧列: Chat Interface
                    gr.ChatInterface(
                        self.echo, additional_inputs=[llm_options_checkbox_group,
                                                      local_private_llm_api,
                                                      local_private_llm_key,
                                                      local_private_llm_name, input_datatable_name,
                                                      input_database_url, input_database_name, input_database_passwd],
                        title="""
            <h1 style='text-align: center; margin-bottom: 1rem; font-family: "Courier New", monospace;
                       background: linear-gradient(135deg, #9400D3, #4B0082, #0000FF, #008000, #FFFF00, #FF7F00, #FF0000);
                       -webkit-background-clip: text;
                       color: transparent;'>
                RainbowSQL-Agent
            </h1>
            """,
                        description="""
                            <style>
                                .footer-email {
                                    position: fixed;
                                    left: 0;
                                    right: 0;
                                    bottom: 0;
                                    text-align: center;
                                    padding: 10px;
                                    background-color: #f8f9fa;
                                    box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
                                    font-family: Arial, sans-serif;
                                    font-size: 14px;
                                }
                                .footer-email a {
                                    color: #007bff;
                                    text-decoration: none;
                                }
                                .footer-email a:hover {
                                    text-decoration: underline;
                                }
                            </style>
                            <div class='footer-email'>
                                <p>How to reach us：<a href='mailto:zhujiadongvip@163.com'>zhujiadongvip@163.com</a></p>
                            </div>
                        """
                    )

    # 创建 Gradio 界面的代码
    def launch(self):
        return self.interface
