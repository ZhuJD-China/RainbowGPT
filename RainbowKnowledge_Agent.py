import queue
import threading
import chromadb
import openai
import datetime
import time
import os
from dotenv import load_dotenv
import gradio as gr
from loguru import logger
# 导入 langchain 模块的相关内容
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.callbacks import FileCallbackHandler
from langchain.chains import LLMChain
from langchain.document_loaders import DirectoryLoader
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import (
    ContextualCompressionRetriever,
    BM25Retriever,
    EnsembleRetriever
)
# Rainbow_utils
from Rainbow_utils.get_tokens_cal_filter import filter_chinese_english_punctuation, num_tokens_from_string, \
    truncate_string_to_max_tokens
from Rainbow_utils import get_google_result
from Rainbow_utils import get_prompt_templates


class RainbowKnowledge_Agent:
    def __init__(self):
        self.load_dotenv()
        self.initialize_variables()
        self.create_interface()

    def load_dotenv(self):
        load_dotenv()

    def initialize_variables(self):
        self.docsearch_db = None
        self.script_name = os.path.basename(__file__)
        self.logfile = "./logs/" + self.script_name + ".log"
        logger.add(self.logfile, colorize=True, enqueue=True)
        self.handler = FileCallbackHandler(self.logfile)
        self.persist_directory = ".chromadb/"
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection_name_select_global = None
        # local private llm name
        self.local_private_llm_name_global = None
        # private llm apis
        self.local_private_llm_api_global = None
        # private llm api key
        self.local_private_llm_key_global = None
        # http proxy
        self.proxy_url_global = None
        # 创建 ChatOpenAI 实例作为底层语言模型
        self.llm = None
        self.llm_name_global = None
        self.embeddings = None
        self.Embedding_Model_select_global = 0
        self.temperature_num_global = 0
        # 文档切分的长度
        self.input_chunk_size_global = None
        # 本地知识库嵌入token max
        self.local_data_embedding_token_max_global = None
        # 在文件顶部定义docsearch_db
        self.docsearch_db = None
        self.human_input_global = None

        self.local_search_template = get_prompt_templates.local_search_template
        self.google_search_template = get_prompt_templates.google_search_template
        # 全局工具列表创建
        self.tools = []
        # memory
        self.agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }
        self.memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
        self.Google_Search_tool = None
        self.Local_Search_tool = None
        self.llm_Agent_checkbox_group = None

    def ask_local_vector_db(self, question):
        if self.llm_name_global == "Private-LLM-Model":
            llm = ChatOpenAI(
                model_name=self.local_private_llm_name_global,
                openai_api_base=self.local_private_llm_api_global,
                openai_api_key=self.local_private_llm_key_global,
                streaming=False,
            )
        else:
            llm = ChatOpenAI(temperature=self.temperature_num_global,
                             openai_api_key=os.getenv('OPENAI_API_KEY'),
                             model=self.llm_name_global)

        local_search_prompt = PromptTemplate(
            input_variables=["combined_text", "human_input", "human_input_first"],
            template=self.local_search_template,
        )
        # 本地知识库工具
        local_chain = LLMChain(
            llm=llm, prompt=local_search_prompt,
            verbose=True,
            # return_final_only=True,  # 指示是否仅返回最终解析的结果
        )

        docs = []
        if self.Embedding_Model_select_global == 0:
            print("OpenAIEmbeddings Search")
            # 结合基础检索器+Embedding上下文压缩
            # 将稀疏检索器（如 BM25）与密集检索器（如嵌入相似性）相结合
            chroma_retriever = self.docsearch_db.as_retriever(search_kwargs={"k": 30})

            # 将压缩器和文档转换器串在一起
            splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
            redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
            relevant_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.76)
            pipeline_compressor = DocumentCompressorPipeline(
                transformers=[splitter, redundant_filter, relevant_filter]
            )
            compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                                   base_retriever=chroma_retriever)
            # compressed_docs = compression_retriever.get_relevant_documents(question, tools=tools)

            the_collection = self.client.get_collection(name=self.collection_name_select_global)
            the_metadata = the_collection.get()
            the_doc_llist = the_metadata['documents']
            bm25_retriever = BM25Retriever.from_texts(the_doc_llist)
            bm25_retriever.k = 30

            # 设置最大尝试次数
            max_retries = 3
            retries = 0
            while retries < max_retries:
                try:
                    # 初始化 ensemble 检索器
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, compression_retriever], weights=[0.5, 0.5]
                    )
                    docs = ensemble_retriever.get_relevant_documents(question)
                    break  # 如果成功执行，跳出循环
                except openai.error.OpenAIError as openai_error:
                    if "Rate limit reached" in str(openai_error):
                        print(f"Rate limit reached: {openai_error}")
                        # 如果是速率限制错误，等待一段时间后重试
                        time.sleep(20)
                        retries += 1
                    else:
                        print(f"OpenAI API error: {openai_error}")
                        docs = []
                        break  # 如果遇到其他错误，跳出循环
            # 处理循环结束后的情况
            if retries == max_retries:
                print(f"Max retries reached. Code execution failed.")
        elif self.Embedding_Model_select_global == 1:
            print("HuggingFaceEmbedding Search")
            chroma_retriever = self.docsearch_db.as_retriever(search_kwargs={"k": 30})
            # docs = chroma_retriever.get_relevant_documents(question)
            # chroma_vectorstore = Chroma.from_texts(the_doc_llist, embeddings)
            # chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 10})

            the_collection = self.client.get_collection(name=self.collection_name_select_global)
            the_metadata = the_collection.get()
            the_doc_llist = the_metadata['documents']
            bm25_retriever = BM25Retriever.from_texts(the_doc_llist)
            bm25_retriever.k = 30

            # 初始化 ensemble 检索器
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
            )
            docs = ensemble_retriever.get_relevant_documents(question)

        cleaned_matches = []
        total_toknes = 0
        last_index = 0
        for index, context in enumerate(docs):
            cleaned_context = context.page_content.replace('\n', ' ').strip()
            cleaned_context = f"{cleaned_context}"
            tokens = num_tokens_from_string(cleaned_context, "cl100k_base")
            # tokens = tokenizers.encode(cleaned_context, add_special_tokens=False)
            if total_toknes + tokens <= (int(self.local_data_embedding_token_max_global)):
                cleaned_matches.append(cleaned_context)
                total_toknes += tokens
            else:
                last_index = index
                break
        print("Embedding了 ", str(last_index + 1), " 个知识库文档块")
        # 将清理过的匹配项组合合成一个字符串
        combined_text = " ".join(cleaned_matches)

        answer = local_chain.predict(combined_text=combined_text, human_input=question,
                                     human_input_first=self.human_input_global)
        return answer

    def get_google_answer(self, question, result_queue):
        google_answer_box = get_google_result.selenium_google_answer_box(
            question, "Rainbow_utils/chromedriver-120.0.6099.56.0.exe")
        # 使用正则表达式保留中文、英文和标点符号
        google_answer_box = filter_chinese_english_punctuation(google_answer_box)
        result_queue.put(("google_answer_box", google_answer_box))

    def process_data_title_summary(self, data_title_Summary, result_queue):
        data_title_Summary_str = ''.join(data_title_Summary)
        result_queue.put(("data_title_Summary_str", data_title_Summary_str))

    def process_custom_search_link(self, custom_search_link, result_queue):
        link_detail_res = []
        for link in custom_search_link[:1]:
            website_content = get_google_result.get_website_content(link)
            if website_content:
                link_detail_res.append(website_content)

        link_detail_string = '\n'.join(link_detail_res)
        link_detail_string = filter_chinese_english_punctuation(link_detail_string)
        result_queue.put(("link_detail_string", link_detail_string))

    def custom_search_and_fetch_content(self, question, result_queue):
        custom_search_link, data_title_Summary = get_google_result.google_custom_search(question)

        # 创建新的线程来处理 data_title_Summary 和 custom_search_link
        thread3 = threading.Thread(target=self.process_data_title_summary, args=(data_title_Summary, result_queue))
        thread4 = threading.Thread(target=self.process_custom_search_link, args=(custom_search_link, result_queue))

        thread3.start()
        thread4.start()
        thread3.join()
        thread4.join()

    def Google_Search_run(self, question):
        get_google_result.set_global_proxy(self.proxy_url_global)

        if self.llm_name_global == "Private-LLM-Model":
            self.llm = ChatOpenAI(
                model_name=self.local_private_llm_name_global,
                openai_api_base=self.local_private_llm_api_global,
                openai_api_key=self.local_private_llm_key_global,
                streaming=False,
            )
        else:
            self.llm = ChatOpenAI(temperature=self.temperature_num_global,
                                  openai_api_key=os.getenv('OPENAI_API_KEY'),
                                  model=self.llm_name_global)

        local_search_prompt = PromptTemplate(
            input_variables=["combined_text", "human_input", "human_input_first"],
            template=self.google_search_template,
        )
        local_chain = LLMChain(
            llm=self.llm, prompt=local_search_prompt,
            verbose=True,
            # return_final_only=True,  # 指示是否仅返回最终解析的结果
        )

        # 创建一个队列来存储线程结果
        results_queue = queue.Queue()
        # 创建并启动线程
        thread1 = threading.Thread(target=self.get_google_answer, args=(question, results_queue))
        thread2 = threading.Thread(target=self.custom_search_and_fetch_content, args=(question, results_queue))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # 初始化变量
        google_answer_box = ""
        data_title_Summary_str = ""
        link_detail_string = ""

        # 提取并分配结果
        while not results_queue.empty():
            result_type, result = results_queue.get()
            if result_type == "google_answer_box":
                google_answer_box = result
            elif result_type == "data_title_Summary_str":
                data_title_Summary_str = result
            elif result_type == "link_detail_string":
                link_detail_string = result

        finally_combined_text = f"""
        当前关键字搜索的答案框数据：
        {google_answer_box}

        搜索结果相似度TOP10的网站的标题和摘要数据：
        {data_title_Summary_str}

        搜索结果相似度TOP1的网站的详细内容数据:
        {link_detail_string}

        """

        truncated_text = truncate_string_to_max_tokens(finally_combined_text,
                                                       self.local_data_embedding_token_max_global,
                                                       "cl100k_base",
                                                       step_size=256)

        answer = local_chain.predict(combined_text=truncated_text, human_input=question,
                                     human_input_first=self.human_input_global)

        return answer

    def echo(self, message, history, llm_options_checkbox_group, collection_name_select,
             temperature_num, print_speed_step, tool_checkbox_group,
             Embedding_Model_select,
             local_data_embedding_token_max,
             Google_proxy, local_private_llm_api, local_private_llm_key,
             local_private_llm_name, llm_Agent_checkbox_group):
        self.human_input_global = message
        self.local_private_llm_name_global = str(local_private_llm_name)
        self.local_private_llm_api_global = str(local_private_llm_api)
        self.local_private_llm_key_global = str(local_private_llm_key)
        self.collection_name_select_global = str(collection_name_select)
        self.local_data_embedding_token_max_global = int(local_data_embedding_token_max)
        self.temperature_num_global = float(temperature_num)
        self.llm_name_global = str(llm_options_checkbox_group)
        self.llm_Agent_checkbox_group = llm_Agent_checkbox_group

        response = (self.llm_name_global + " 模型加载中....." + "temperature="
                    + str(self.temperature_num_global))
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]
        if self.llm_name_global == "Private-LLM-Model":
            self.llm = ChatOpenAI(
                model_name=self.local_private_llm_name_global,
                openai_api_base=self.local_private_llm_api_global,
                openai_api_key=self.local_private_llm_key_global,
                streaming=False,
            )
        else:
            self.llm = ChatOpenAI(temperature=self.temperature_num_global,
                                  openai_api_key=os.getenv('OPENAI_API_KEY'),
                                  model=self.llm_name_global)

        self.tools = []  # 重置工具列表
        # Check if 'wolfram-alpha' is in the selected tools
        if "wolfram-alpha" in tool_checkbox_group:
            # Load both 'wolfram-alpha' and 'arxiv' tools
            self.tools = load_tools(["wolfram-alpha", "arxiv"], llm=self.llm)
        else:
            # Load only the 'arxiv' tool
            self.tools = load_tools(["arxiv"], llm=self.llm)

        self.Google_Search_tool = Tool(
            name="Google_Search",
            func=self.Google_Search_run,
            description="""
            如果本地知识库中无答案或问题需要网络搜索可用这个互联网搜索工具进行搜索问答。
            1.你先根据我的问题提取出最适合Google搜索引擎搜索的关键字进行搜索,可以选择英语或者中文搜索
            2.同时增加一些搜索提示词包括(使用引号，时间范围，关键字和符号)
            3.如果问题比较复杂，你可以一步一步的思考去搜索和回答
            """
        )
        self.Local_Search_tool = Tool(
            name="Local_Search",
            func=self.ask_local_vector_db,
            description="""
                这是一个本地知识库搜索工具，你可以优先使用本地搜索并总结回答。
                1.你先根据我的问题提取出最适合embedding模型向量匹配的关键字进行搜索。
                2.注意你需要提出非常有针对性准确的问题和回答。
                3.如果问题比较复杂，可以将复杂的问题进行拆分，你可以一步一步的思考。
                """
        )

        # Initialize flags for additional tools
        flag_get_Local_Search_tool = False
        # Check for additional tools and append them if not already in the list
        for tg in tool_checkbox_group:
            if tg == "Google Search" and self.Google_Search_tool not in self.tools:
                response = "Google Search 工具加入 回答中..........."
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[:i + int(print_speed_step)]

                self.tools.append(self.Google_Search_tool)
                self.proxy_url_global = str(Google_proxy)  # Set proxy
                os.environ['http_proxy'] = self.proxy_url_global
                os.environ['https_proxy'] = self.proxy_url_global

            elif tg == "Local Knowledge Search" and self.Local_Search_tool not in self.tools:
                response = "Local Knowledge Search 工具加入 回答中..........."
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[:i + int(print_speed_step)]
                self.tools.append(self.Local_Search_tool)

                response = (f"{self.llm_name_global} & {Embedding_Model_select} 模型加载中.....temperature="
                            + str(self.temperature_num_global))
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[:i + int(print_speed_step)]

                if Embedding_Model_select in ["Openai Embedding", "", None]:
                    self.embeddings = OpenAIEmbeddings()
                    self.embeddings.show_progress_bar = True
                    self.embeddings.request_timeout = 20
                    self.Embedding_Model_select_global = 0
                elif Embedding_Model_select == "HuggingFace Embedding":
                    self.embeddings = HuggingFaceEmbeddings(cache_folder="models")
                    self.Embedding_Model_select_global = 1

                flag_get_Local_Search_tool = True

        if message == "":
            response = "哎呀！好像有点小尴尬，您似乎忘记提出问题了。别着急，随时输入您的问题，我将尽力为您提供帮助！"
            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]
            return

        if flag_get_Local_Search_tool:
            if collection_name_select and collection_name_select != "...":
                print(f"{collection_name_select}", " Collection exists, load it")
                response = f"{collection_name_select}" + "知识库加载中，请等待我的回答......."
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[: i + int(print_speed_step)]
                self.docsearch_db = Chroma(client=self.client, embedding_function=self.embeddings,
                                           collection_name=collection_name_select)
            else:
                response = "未选择知识库，回答中止。"
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[: i + int(print_speed_step)]
                return
        if not flag_get_Local_Search_tool:
            response = "没有打开本地知识库搜索工具，使用模型记忆或其他工具回答。"
            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]

        # 初始化agent代理
        agent_open_functions = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=self.llm_Agent_checkbox_group,
            verbose=True,
            agent_kwargs=self.agent_kwargs,
            memory=self.memory,
            max_iterations=10,
            early_stopping_method="generate",
            handle_parsing_errors=True,  # 初始化代理并处理解析错误
            callbacks=[self.handler],
        )
        try:
            response = agent_open_functions.run(message)
        except Exception as e:
            response = f"发生错误：{str(e)}"
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]
        # response = agent_open_functions.run(message)
        # return response

    def update_collection_name(self):
        # 获取已存在的collection的名称列表
        collections = self.client.list_collections()
        collection_name_choices = []
        for collection in collections:
            collection_name = collection.name
            collection_name_choices.append(collection_name)
        # 调用gr.Dropdown.update方法，传入新的选项列表
        return gr.Dropdown.update(choices=collection_name_choices)

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
                            local_private_llm_name = gr.Textbox(value="Qwen-*B-Chat", label="Private llm name")
                            llm_Agent = ["chat-zero-shot-react-description", "openai-functions",
                                         "zero-shot-react-description"]
                            llm_Agent_checkbox_group = gr.Dropdown(llm_Agent, label="LLM Agent Type Options",
                                                                   value=llm_Agent[0])

                        with gr.Group():
                            gr.Markdown("### Private LLM Settings")
                            local_private_llm_api = gr.Textbox(value="http://172.16.0.160:8000/v1",
                                                               label="Private llm openai-api base")
                            local_private_llm_key = gr.Textbox(value="EMPTY", label="Private llm openai-api key")
                            Google_proxy = gr.Textbox(value="http://localhost:7890", label="System Http Proxy")

                    with gr.Row():
                        with gr.Group():
                            gr.Markdown("### Additional Tools")

                            tool_options = ["Google Search", "Local Knowledge Search", "wolfram-alpha"]
                            tool_checkbox_group = gr.CheckboxGroup(tool_options, label="Tools Select")
                            gr.Markdown("Note: 'arxiv' Tools are enabled by default.")

                        with gr.Group():
                            gr.Markdown("### Knowledge Collection Settings")
                            collection_name_select = gr.Dropdown(["..."], label="Select existed Collection",
                                                                 value="...")
                            Refresh_button = gr.Button("Refresh Collection", variant="secondary")
                            Refresh_button.click(fn=self.update_collection_name, outputs=collection_name_select)

                            temperature_num = gr.Slider(0, 1, label="Temperature")
                            print_speed_step = gr.Slider(10, 20, label="Print Speed Step", step=1)

                    with gr.Group():
                        gr.Markdown("### Embedding Data Settings")
                        Embedding_Model_select = gr.Radio(["Openai Embedding", "HuggingFace Embedding"],
                                                          label="Embedding Model Select",
                                                          value="HuggingFace Embedding")
                        local_data_embedding_token_max = gr.Slider(1024, 15360, step=2,
                                                                   label="Embeddings Data Max Tokens",
                                                                   value=2048)
                with gr.Column(scale=5):
                    # 右侧列: Chat Interface
                    gr.ChatInterface(
                        self.echo, additional_inputs=[llm_options_checkbox_group, collection_name_select,
                                                      temperature_num, print_speed_step, tool_checkbox_group,
                                                      Embedding_Model_select,
                                                      local_data_embedding_token_max,
                                                      Google_proxy, local_private_llm_api, local_private_llm_key,
                                                      local_private_llm_name, llm_Agent_checkbox_group],
                        title="""
            <h1 style='text-align: center; margin-bottom: 1rem; font-family: "Courier New", monospace;
                       background: linear-gradient(135deg, #9400D3, #4B0082, #0000FF, #008000, #FFFF00, #FF7F00, #FF0000);
                       -webkit-background-clip: text;
                       color: transparent;'>
                RainbowGPT-Agent
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

    def launch(self):
        return self.interface
