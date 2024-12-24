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
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain


class VerboseHandler(BaseCallbackHandler):
    def __init__(self):
        self.steps = []
        self.current_iteration = 0
        super().__init__()
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.steps.append("🤖 开始分析SQL查询...")
    
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'generations'):
            for gen in response.generations:
                if gen:
                    self.steps.append(f"🤔 思考结果: {gen[0].text}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown tool")
        self.steps.append(f"🔧 开始执行: {tool_name}")
        if "sql" in tool_name.lower():
            self.steps.append(f"📝 SQL语句:\n```sql\n{input_str}\n```")
    
    def on_agent_action(self, action, color=None, **kwargs):
        try:
            self.current_iteration += 1
            self.steps.append(f"\n**第 {self.current_iteration} 轮执行**")
            
            if hasattr(action, 'log'):
                self.steps.append(f"**思考过程:** {action.log}")
            
            if hasattr(action, 'tool'):
                self.steps.append(f"**执行操作:** {action.tool}")
            
            if hasattr(action, 'tool_input'):
                self.steps.append(f"**输入参数:**\n```sql\n{action.tool_input}\n```")
        except Exception as e:
            self.steps.append(f"**注意:** 操作记录出现问题: {str(e)}")
    
    def on_agent_observation(self, observation, color=None, **kwargs):
        try:
            if observation:
                self.steps.append(f"**观察结果:**\n```\n{observation}\n```")
        except Exception as e:
            self.steps.append(f"**注意:** 结果记录出现问题: {str(e)}")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        self.steps.append("🔄 开始执行查询链...")
    
    def on_chain_end(self, outputs, **kwargs):
        self.steps.append("✅ 查询链执行完成")
    
    def on_chain_error(self, error, **kwargs):
        self.steps.append(f"❌ 执行出错: {str(error)}")
    
    def on_agent_finish(self, finish, color=None, **kwargs):
        try:
            if hasattr(finish, 'log'):
                self.steps.append(f"\n**执行总结**")
                self.steps.append(f"**结论:** {finish.log}")
            
            if hasattr(finish, 'return_values'):
                output = finish.return_values.get('output', str(finish.return_values))
                self.steps.append(f"**最终结果:** {output}")
        except Exception as e:
            self.steps.append(f"**注意:** 完成记录出现问题: {str(e)}")
    
    def get_steps(self):
        return "\n".join(self.steps)


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

        self.intermediate_handler = VerboseHandler()

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

        try:
            # 创建 verbose handler
            verbose_handler = VerboseHandler()
            
            # 创建SQL工具包
            toolkit = SQLDatabaseToolkit(
                db=db,
                llm=llm,
                use_query_checker=True
            )

            # 获取工具列表
            tools = toolkit.get_tools()

            # 创建代理提示
            prefix = """你是一个能帮助用户操作SQL数据库的智能助手。在回答问题时，请遵循以下思考步骤：

1. 仔细分析用户的问题，理解需求
2. 检查数据库结构，确定需要查询或修改的表
3. 设计合适的SQL语句
4. 执行操作并验证结果

请严格按照以下格式回复：

Thought: 分析问题并说明思考过程
Action: 选择要使用的工具（sql_db_query, sql_db_schema, sql_db_list_tables）
Action Input: 具体的SQL查询或命令
Observation: 工具返回的结果
... (根据需要重复上述步骤)
Thought: 总结所有信息
Final Answer: 给出完整的答案

当前可用的工具有:"""

            suffix = """请记住：
1. 在执行修改操作前，先确认当前数据状态
2. 确保SQL语句准确无误
3. 验证操作结果
4. 给出清晰的解释

当前问题: {input}
思考过程: {agent_scratchpad}"""

            # 创建代理
            from langchain.agents import ZeroShotAgent, AgentExecutor
            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "agent_scratchpad"]
            )

            llm_chain = LLMChain(llm=llm, prompt=prompt)
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=5,
                handle_parsing_errors=True,
                callbacks=[verbose_handler],
                return_intermediate_steps=True
            )

            # 执行查询
            result = agent_executor(
                {
                    "input": f"""本次交易说明：
                    1. 你有完整的数据库修改权限
                    2. 你应该直接执行用户的请求
                    3. 修改后，验证并报告结果,特别的你需要指出数据库变化的地方查询展示。
                    
                    用户请求：{message}"""
                }
            )

            # 获取执行步骤和结果
            execution_steps = verbose_handler.get_steps()
            final_response = result["output"] if isinstance(result, dict) and "output" in result else str(result)
            
            # 构建完整响应
            full_response = f"""
### 执行过程
{execution_steps}

### 最终结果
{final_response}
"""

            # 记录日志
            logger.info(f"User Input: {message}")
            logger.info(f"Execution Steps: {execution_steps}")
            logger.info(f"Final Response: {final_response}")

            # 逐步显示响应
            for i in range(0, len(full_response), int(print_speed_step)):
                yield full_response[: i + int(print_speed_step)]

        except Exception as e:
            error_msg = f"发生错误：{str(e)}"
            logger.error(error_msg)
            yield error_msg

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as self.interface:
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    # 左侧列: 所有控件
                    with gr.Row():
                        with gr.Group():
                            gr.Markdown("### Language Model Selection")
                            llm_options = ["gpt-4o", "Private-LLM-Model"]
                            llm_options_checkbox_group = gr.Dropdown(llm_options, label="LLM Model Select Options",
                                                                     value=llm_options[0])
                            local_private_llm_name = gr.Textbox(value="gpt-4o-mini", label="Private llm name")

                        with gr.Group():
                            gr.Markdown("### Private LLM Settings")
                            local_private_llm_api = gr.Textbox(value="https://api.chatanywhere.tech",
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
                                value=None  # 初始为 None
                            )
                            update_button = gr.Button("Update Tables List")
                            update_button.click(fn=self.update_tables_list,
                                                inputs=[input_database_url, input_database_name,
                                                        input_database_passwd],
                                                outputs=input_datatable_name)
                with gr.Column(scale=5):
                    # 右侧列: Chat Interface
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
                        additional_inputs=[llm_options_checkbox_group,
                                           local_private_llm_api,
                                           local_private_llm_key,
                                           local_private_llm_name, input_datatable_name,
                                           input_database_url, input_database_name, input_database_passwd],
                        title="RainbowSQL-Agent",
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

    # 创建 Gradio 界面的代码
    def launch(self):
        return self.interface

    # 创建 Gradio 界面的代码
    def launch(self):
        return self.interface
