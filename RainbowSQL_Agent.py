import os
from dotenv import load_dotenv
import gradio as gr
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.callbacks import FileCallbackHandler
from langchain_core.prompts import MessagesPlaceholder
# 导入 langchain 模块的相关内容
from sqlalchemy import create_engine
# Rainbow_utils
from loguru import logger
from urllib.parse import quote_plus


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
            print(host, username, password)
            # 对密码进行URL编码
            encoded_password = quote_plus(password)
            # 构建数据库连接字符串
            connection_string = f"mysql+pymysql://{username}:{encoded_password}@{host}/"
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
        databases = self.get_database_tables(host, username, password)
        return gr.Dropdown(
            choices=databases,
            value=databases[0] if databases else None
        )

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
            response = "哎呀！好像有点小尴，您似乎忘记提出问题了。别着急，随时输入您的问题，我将尽力为您提供帮助！"
            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]
            return

        db_name = input_datatable_name
        # 对密码进行URL编码
        encoded_password = quote_plus(input_database_passwd)
        # 创建数据库连接
        db = SQLDatabase.from_uri(
            f"mysql+pymysql://{input_database_name}:{encoded_password}@{input_database_url}/{db_name}",
            # include_tables=['inventory_check', 'inventory_details', 'products'],  # 明确指定表
            sample_rows_in_table_info=3,
            view_support=True
        )

        # 创建代理执行器
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(
                db=db, 
                llm=llm, 
                use_query_checker=False
            ),
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            agent_kwargs={
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
                "prefix": """你是一个能帮助用户操作SQL数据库的智能助手。
                            你拥有执行所有SQL命令的完整权限，包括 SELECT、INSERT、UPDATE 和 DELETE。
                            当被要求修改数据时，你应该：
                            1. 首先确认数据的当前状态
                            2. 执行所需的修改操作
                            3. 验证修改是否成功完成
                            
                            重要提示：你被允许且应该在用户要求时执行数据修改操作。""",
                "format_instructions": """在进行数据修改时，你应该始终以SQL语句系列的形式输出。
                                       首先说明你计划做什么，然后执行它。"""
            },
            memory=self.memory,
            max_iterations=5,
            callbacks=[self.handler],
        )

        try:
            response = agent_executor.run(
                f"""本次交��说明：
                1. 你有完整的数据库修改权限
                2. 你应该直接执行用户的请求
                3. 修改后，验证并报告结果,特别的你需要指出数据库变化的地方查询展示。
                
                用户请求：{message}"""
            )
        except Exception as e:
            response = f"发生错误：{str(e)}"
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]
        logger.info(response)

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
                            input_datatable_name = gr.Dropdown(
                                choices=[],  # 初始为空列表
                                label="Database Select Name",
                                value=None  # 初始值为 None
                            )
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
