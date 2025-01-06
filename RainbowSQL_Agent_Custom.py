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
from sqlalchemy import create_engine
from loguru import logger
from urllib.parse import quote_plus
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from Rainbow_utils.model_config_manager import ModelConfigManager
from datetime import datetime
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from typing import List


class RainbowSQLAgentCustom:
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
        
        # 初始化模型配置管理器
        self.model_manager = ModelConfigManager()
        
        self.human_input_global = None
        self.agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }
        self.memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    def get_llm(self):
        """获取当前配置的LLM实例"""
        config = self.model_manager.get_active_config()
        return ChatOpenAI(
            model_name=config.model_name,
            openai_api_base=config.api_base,
            openai_api_key=config.api_key,
            temperature=config.temperature,
            streaming=True
        )

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

    # 函数用于更新下拉列表的表格名称
    def update_tables_list(self, host, username, password):
        databases = self.get_database_tables(host, username, password)
        return gr.Dropdown(
            choices=databases,
            value=databases[0] if databases else None
        )

    def echo(self, message, history, input_datatable_name,
             input_database_url, input_database_name, input_database_passwd):
        """
        处理用户查询的主要方法
        """
        print_speed_step = 10
        self.human_input_global = message

        # 获取当前配置的LLM
        llm = self.get_llm()

        if message == "":
            response = "哎呀！好像有点小尴尬，您似乎忘记提出问题了。别着急，随时输入您的问题，我将尽力为您提供帮助！"
            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]
            return

        try:
            # 1. 建立数据库连接
            db_name = input_datatable_name
            encoded_password = quote_plus(input_database_passwd)
            db = SQLDatabase.from_uri(
                f"mysql+pymysql://{input_database_name}:{encoded_password}@{input_database_url}/{db_name}",
                sample_rows_in_table_info=3,
                view_support=True
            )
            tables = db.get_usable_table_names()
            print("可用表格:", tables)
            
            if not tables:
                yield "数据库中没有可用的表格。"
                return
            
            # 2. 第一步：选择相关表格
            table_selector_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个数据库专家。根据用户的问题，从给定的表格列表中选择相关的表格。
                请仔细分析用户问题，只选择必要的表格。
                
                重要提示：
                - 必须从提供的表格列表中选择
                - 返回完整的表格名称，不要拆分或修改
                - 用英文逗号分隔多个表格
                - 如果不确定，返回所有表格名称
                - 确保返回的表格名称完全匹配原始表格名称
                
                示例输出格式：
                table1,table2,table3"""),
                ("human", "可用的表格列表: {tables}"),
                ("human", "用户问题: {question}\n请列出相关表格名称:")
            ])
            
            table_chain = LLMChain(
                llm=llm,
                prompt=table_selector_prompt
            )
            
            # 获取表格选择结果并处理
            table_selection = table_chain.run(
                tables=", ".join(tables), 
                question=message
            ).strip()
            print("模型返回的表格选择:", table_selection)
            
            # 改进的表格名称清理和验证逻辑
            selected_tables = []
            for table in table_selection.split(','):
                # 移除任何引号或其他特殊字符
                cleaned_table = table.strip().strip('"\'`[] ')
                if cleaned_table in tables:  # 严格匹配
                    selected_tables.append(cleaned_table)
            
            # 如果没有选择到有效表格，使用所有表格
            if not selected_tables:
                print("没有找到有效的表格，使用所有表格")
                selected_tables = tables
            
            print("最终选择的表格:", selected_tables)
            yield f"已选择相关表格: {', '.join(selected_tables)}\n\n"
            selected_tables = list(set(selected_tables))
            
            # 3. 第二步：生成SQL查询
            # 获取表格结构信息
            table_info = "\n".join([
                f"表格 {table}:\n{db.get_table_info(table)}\n"
                for table in selected_tables
            ])
            print("table_info",table_info)

            sql_generator_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个SQL专家。请根据用户问题和选定的表格生成SQL查询。
                要求：
                1. 只生成有效的SQL查询语句
                2. 只使用提供的表格
                3. 确保SQL语法正确
                4. 不要包含任何注释或解释
                5. 使用适当的JOIN操作（如果需要）"""),
                ("human", """可用表格及其结构:
                {table_info}
                
                用户问题: {question}
                
                请生成SQL查询:""")
            ])
            
            print("sql_generator_prompt",sql_generator_prompt)

            sql_chain = LLMChain(
                llm=llm,
                prompt=sql_generator_prompt
            )
            
            sql_query = sql_chain.run(
                table_info=table_info,
                question=message
            ).strip()
            print("sql_query",sql_query)
            
            yield f"生成的SQL查询:\n```sql\n{sql_query}\n```\n\n"
            
            # 4. 执行SQL查询并格式化结果
            result = db.run(sql_query)
            
            # 5. 使用第三个模型解释结果
            result_explainer_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个数据分析专家。请用通俗易懂的语言解释SQL查询的结果。
                要求：
                1. 清晰解释查询结果的含义
                2. 突出重要的发现
                3. 使用易懂的语言
                4. 如果结果为空，说明可能的原因"""),
                ("human", """SQL查询: {sql}
                查询结果: {result}
                
                请解释这个结果:""")
            ])
            
            explain_chain = LLMChain(
                llm=llm,
                prompt=result_explainer_prompt
            )
            
            explanation = explain_chain.run(sql=sql_query, result=result)
            
            # 6. 返回完整的响应
            full_response = f"{explanation}\n\n原始结果:\n{result}"
            
            # 逐步显示响应
            for i in range(0, len(full_response), int(print_speed_step)):
                yield full_response[: i + int(print_speed_step)]

        except Exception as e:
            error_msg = f"发生错误：{str(e)}"
            logger.error(error_msg)
            yield error_msg

    def test_connection(self, host, username, password):
        """测试数据库连接"""
        try:
            encoded_password = quote_plus(password)
            connection_string = f"mysql+pymysql://{username}:{encoded_password}@{host}/"
            engine = create_engine(connection_string)
            connection = engine.connect()
            connection.close()
            return "✅ 连接成功", "已连接"
        except Exception as e:
            return f"❌ 连接失败: {str(e)}", "连接失败"

    def refresh_connection(self, host, username, password, current_db):
        """刷新数据库连接"""
        try:
            databases = self.get_database_tables(host, username, password)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if current_db:
                # 获取当前数据库的表数量
                encoded_password = quote_plus(password)
                connection_string = f"mysql+pymysql://{username}:{encoded_password}@{host}/{current_db}"
                engine = create_engine(connection_string)
                with engine.connect() as conn:
                    result = conn.execute("SHOW TABLES")
                    table_count = len(list(result))
            else:
                table_count = 0
            
            db_info = {
                "已选数据库": current_db or "无",
                "表数量": table_count,
                "连接时间": current_time
            }
            
            return gr.Dropdown(choices=databases, value=current_db), db_info, "已连接"
        except Exception as e:
            return None, {"错误": str(e)}, "连接失败"

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as self.interface:
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Group():
                            gr.Markdown("""
                            ### 🌈 Database Settings
                            
                            #### 使用说明
                            1. 📝 填写数据库连接信息
                               - Database URL: 数据库服务器地址（默认localhost）
                               - Username: 数据库用户名（默认root）
                               - Password: 数据库密码
                            
                            2. 🔄 点击"Update Tables List"更新数据库列表
                            
                            3. 📊 从下拉菜单选择要操作的数据库
                            
                            #### 连接状态
                            """)
                            
                            # 添加连接状态指示器
                            connection_status = gr.Textbox(
                                value="未连接",
                                label="数据库连接状态",
                                interactive=False,
                                container=False,
                            )
                            
                            # 数据库连接信息分组
                            with gr.Group():
                                gr.Markdown("#### 📊 数据库连接信息")
                                input_database_url = gr.Textbox(
                                    value="localhost",
                                    label="Database URL",
                                    placeholder="例如: localhost 或 127.0.0.1"
                                )
                                with gr.Row():
                                    input_database_name = gr.Textbox(
                                        value="root",
                                        label="Username",
                                        placeholder="数据库用户名"
                                    )
                                    input_database_passwd = gr.Textbox(
                                        value="",
                                        label="Password",
                                        type="password",
                                        placeholder="数据库密码"
                                    )
                                
                                # 添加测试连接按钮
                                test_connection_btn = gr.Button(
                                    "🔍 测试连接",
                                    variant="secondary"
                                )
                            
                            # 数据库选择分组
                            with gr.Group():
                                gr.Markdown("#### 📁 数据库选择")
                                input_datatable_name = gr.Dropdown(
                                    choices=[],
                                    label="选择数据库",
                                    value=None,
                                    container=True,
                                    interactive=True
                                )
                                with gr.Row():
                                    update_button = gr.Button(
                                        "🔄 更新数据库列表",
                                        variant="primary"
                                    )
                                    refresh_btn = gr.Button(
                                        "🔄 刷新连接",
                                        variant="secondary"
                                    )
                            
                            # 添加数据库信息显示区
                            with gr.Group():
                                gr.Markdown("#### ℹ️ 当前数据库信息")
                                db_info = gr.JSON(
                                    value={
                                        "已选数据库": "无",
                                        "表数量": 0,
                                        "连接时间": "未连接"
                                    },
                                    label="数据库详情"
                                )
                            
                            # 添加结果显示区
                            connection_result = gr.Textbox(
                                label="连接测试结果",
                                visible=False
                            )

                            # 绑定按钮事件
                            test_connection_btn.click(
                                fn=self.test_connection,
                                inputs=[
                                    input_database_url,
                                    input_database_name,
                                    input_database_passwd
                                ],
                                outputs=[
                                    connection_result,
                                    connection_status
                                ]
                            )
                            
                            update_button.click(
                                fn=self.update_tables_list,
                                inputs=[
                                    input_database_url,
                                    input_database_name,
                                    input_database_passwd
                                ],
                                outputs=[input_datatable_name]
                            )
                            
                            refresh_btn.click(
                                fn=self.refresh_connection,
                                inputs=[
                                    input_database_url,
                                    input_database_name,
                                    input_database_passwd,
                                    input_datatable_name
                                ],
                                outputs=[
                                    input_datatable_name,
                                    db_info,
                                    connection_status
                                ]
                            )
                            
                            # 数据库选择变化时更新信息
                            input_datatable_name.change(
                                fn=self.refresh_connection,
                                inputs=[
                                    input_database_url,
                                    input_database_name,
                                    input_database_passwd,
                                    input_datatable_name
                                ],
                                outputs=[
                                    input_datatable_name,
                                    db_info,
                                    connection_status
                                ]
                            )

                with gr.Column(scale=5):
                    # Chat Interface部分
                    custom_css = """
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

                            /* 调整聊天界面容器高度 */
                            .gradio-container {
                                min-height: 95vh !important;
                            }
                            
                            /* 调整聊天记录区域高度 */
                            .chat-history {
                                height: calc(95vh - 200px) !important;
                                overflow-y: auto;
                            }
                            
                            /* 标题样式 */
                            .gradio-header h1 {
                                text-align: center;
                                margin-bottom: 1rem;
                                font-family: "Courier New", monospace;
                                background: linear-gradient(135deg, #9400D3, #4B0082, #0000FF, #008000, #FFFF00, #FF7F00, #FF0000);
                                -webkit-background-clip: text;
                                -webkit-text-fill-color: transparent;
                                background-clip: text;
                                color: transparent;
                            }
                        </style>
                    """

                    gr.ChatInterface(
                        self.echo,
                        additional_inputs=[input_datatable_name,
                                       input_database_url, input_database_name, input_database_passwd],
                        title="RainbowSQL-Agent Custom",
                        css=custom_css,
                        description="""
                            <div class='footer-email'>
                                <p>How to reach us：<a href='mailto:zhujiadongvip@163.com'>zhujiadongvip@163.com</a></p>
                            </div>
                        """,
                        theme="soft",
                        fill_height=True,
                        autoscroll=True
                    )

    def launch(self):
        return self.interface
