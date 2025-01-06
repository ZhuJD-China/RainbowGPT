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
            
            # 修改表格选择和验证逻辑
            selected_tables = []
            for table in table_selection.split(','):
                # 移除任何引号、空格和其他特殊字符
                cleaned_table = table.strip().strip('"\'`[] ')
                if cleaned_table in tables:  # 严格匹配
                    selected_tables.append(cleaned_table)
                else:
                    print(f"警告: 表格 '{cleaned_table}' 在数据库中未找到")
            
            # 如果没有选择到有效表格，使用所有表格
            if not selected_tables:
                print("没有找到有效的表格，使用所有表格")
                selected_tables = tables
            else:
                # 确保表格名称列表中没有重复
                selected_tables = list(dict.fromkeys(selected_tables))
            
            print("最终选择的表格:", selected_tables)
            yield f"已选择相关表格: {', '.join(selected_tables)}\n\n"

            # 获取表格结构信息
            table_info_list = []
            for table in selected_tables:
                try:
                    query = f"""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE,
                        COLUMN_KEY,
                        COLUMN_COMMENT
                    FROM 
                        INFORMATION_SCHEMA.COLUMNS 
                    WHERE 
                        TABLE_SCHEMA = '{db_name}' 
                        AND TABLE_NAME = '{table}'
                    """
                    result = db.run(query)
                    table_info_list.append(f"表格 {table}:\n{result}")
                except Exception as e:
                    print(f"获取表格 {table} 信息时出错: {str(e)}")
                    continue
            
            table_info = "\n".join(table_info_list)
            print("table_info", table_info)

            # 3. 生成SQL查询
            sql_generator_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的SQL专家。请根据用户问题生成最优的SQL查询语句。

                关键要求：
                1. 只输出纯SQL代码，不要包含任何注释或其他格式
                2. 只使用提供的表格和字段
                3. 确保SQL语法正确且性能优化
                4. 优先使用内连接(INNER JOIN)，必要时使用左连接(LEFT JOIN)
                
                查询优化准则：
                1. 模糊匹配处理：
                   - 文本搜索时使用 LIKE '%关键词%' 或 REGEXP 进行模糊匹配
                   - 必要时使用 LOWER() 或 UPPER() 实现大小写不敏感匹配
                   - 考虑使用 SOUNDEX() 或 LEVENSHTEIN 处理相似发音词
                
                2. 数值处理：
                   - 使用 BETWEEN 处理范围查询
                   - 使用 ROUND(), CEIL(), FLOOR() 处理数值精度
                   - 注意 NULL 值的处理，使用 COALESCE() 或 IFNULL() 提供默认值
                
                3. 日期时间处理：
                   - 使用 DATE_FORMAT() 格式化日期
                   - 使用 DATEDIFF(), TIMEDIFF() 计算时间差
                   - 使用 DATE_ADD(), DATE_SUB() 进行日期计算
                
                4. 结果优化：
                   - 使用 DISTINCT 去除重复结果
                   - 合理使用 GROUP BY 和聚合函数
                   - 使用 ORDER BY 对结果进行排序
                   - 必要时使用 LIMIT 限制结果数量
                
                5. 性能优化：
                   - 避免使用 SELECT *
                   - 合理使用索引字段
                   - 优先使用 EXISTS 代替 IN
                   - 大数据量时注意分页查询
                
                示例输出格式：
                SELECT column1, column2 FROM table1 
                INNER JOIN table2 ON table1.id = table2.id 
                WHERE column1 LIKE '%keyword%';"""),
                ("human", """可用表格及其结构:
                {table_info}
                
                用户问题: {question}
                
                生成SQL查询:""")
            ])
            
            # print("sql_generator_prompt",sql_generator_prompt)

            sql_chain = LLMChain(
                llm=llm,
                prompt=sql_generator_prompt
            )
            
            sql_query = sql_chain.run(
                table_info=table_info,
                question=message
            ).strip()
            
            # 清理SQL查询，移除可能的markdown标记和多余的空行
            sql_query = (sql_query
                        .replace('```sql', '')
                        .replace('```', '')
                        .strip())
            
            # 记录原始SQL查询用于显示
            display_sql = sql_query
            print("display_sql", display_sql)

            # 显示给用户的SQL查询（使用markdown格式）
            yield f"生成的SQL查询:\n```sql\n{display_sql}\n```\n\n"
            
            # 4. 执行SQL查询并格式化结果
            try:
                result = db.run(sql_query)
                print("执行SQL查询并格式化结果: ", result)
            except Exception as e:
                error_msg = f"SQL执行错误：{str(e)}"
                print(error_msg)
                yield error_msg
                return
            
            # 5. 基于查询结果进行智能问答
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个基于数据的智能问答助手。根据数据库查询结果回答用户问题。
                要求：
                1. 直接回答用户的问题
                2. 使用数据支持你的回答
                3. 如果数据不足以完整回答问题，请说明原因
                4. 保持回答准确、简洁
                5. 可以根据数据做出合理的推断，但要说明这是推断"""),
                ("human", """用户问题: {question}
                查询结果: {result}
                
                请根据数据回答问题:""")
            ])
            # print("qa_prompt",qa_prompt)

            qa_chain = LLMChain(
                llm=llm,
                prompt=qa_prompt
            )
            
            
            answer = qa_chain.run(question=message, result=result)
            
            # 修改这里：截取结果数据
            MAX_RESULT_LENGTH = 500  # 设置最大显示长度
            truncated_result = result
            if len(result) > MAX_RESULT_LENGTH:
                truncated_result = result[:MAX_RESULT_LENGTH] + "...(更多数据已省略)"
            
            # 返回截断后的完整响应
            full_response = f"{answer}\n\n参考数据:\n{truncated_result}"
            
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
