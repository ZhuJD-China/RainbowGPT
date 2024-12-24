import queue
import threading
import chromadb
import openai
import time
import os
from dotenv import load_dotenv
import gradio as gr
from langchain import hub
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents.format_scratchpad import format_log_to_str, format_to_openai_function_messages
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser, OpenAIFunctionsAgentOutputParser
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.utilities import WolframAlphaAPIWrapper, ArxivAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import FileCallbackHandler
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, render_text_description
from langchain_core.utils.function_calling import format_tool_to_openai_function
from langchain_text_splitters import CharacterTextSplitter
from loguru import logger
from langchain.chains import LLMMathChain
from langchain.callbacks.base import BaseCallbackHandler

# Rainbow_utils
from Rainbow_utils.get_tokens_cal_filter import filter_chinese_english_punctuation, num_tokens_from_string, \
    truncate_string_to_max_tokens, concatenate_if_dissimilar
from Rainbow_utils import get_google_result
from Rainbow_utils import get_prompt_templates
from Rainbow_utils.image_genearation import ImageGen


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
        logger.add(self.logfile,
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                   level="DEBUG",
                   rotation="1 MB",
                   compression="zip"
                   )
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
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            output_key="output"
        )

        self.Google_Search_tool = None
        self.Local_Search_tool = None
        self.llm_Agent_checkbox_group = None
        self.intermediate_steps_log = ""

        # 替换为 LLMMathChain
        llm_math = LLMMathChain.from_llm(llm=ChatOpenAI(temperature=0))
        self.math_tool = Tool(
            name="Calculator",
            func=llm_math.run,
            description="""
                这是一个数学计算工具。当你需要:
                1. 执行基础数学运算（加减乘除）
                2. 处理复杂数学表达式
                3. 解决数学问题
                4. 进行数值计算
                使用这个工具时，请提供清晰的数学表达式。
                输入应该是一个数学问题，用自然语言描述。
            """
        )

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

            # 将缩器和文档转换器串在一起
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

            # 置次数
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

    def createImageByBing(self, input):
        auth_cooker = os.getenv('BINGCOKKIE')
        sync_gen = ImageGen(auth_cookie=auth_cooker)
        image_list = sync_gen.get_images(input)
        response = []
        if image_list is None:
            return "我无法为您生成对应的图片，请重试或者补充您的描述"
        else:
            for url in image_list:
                if not url.endswith(".svg"):
                    response.append(url)
            return response

    def get_google_answer(self, question, result_queue):
        try:
            logger.debug("Starting get_google_answer")
            google_answer_box = get_google_result.selenium_google_answer_box(
                question, "Rainbow_utils/chromedriver.exe")
            google_answer_box = filter_chinese_english_punctuation(google_answer_box)
            result_queue.put(("google_answer_box", google_answer_box))
            logger.debug("Completed get_google_answer successfully")
        except Exception as e:
            logger.exception(f"Error in get_google_answer: {str(e)}")
            result_queue.put(("google_answer_box", f"获取Google答案时出错: {str(e)}"))

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
        try:
            logger.debug("Starting custom_search_and_fetch_content")
            custom_search_link, data_title_Summary = get_google_result.google_custom_search(question)

            thread3 = threading.Thread(target=self.process_data_title_summary,
                                       args=(data_title_Summary, result_queue))
            thread4 = threading.Thread(target=self.process_custom_search_link,
                                       args=(custom_search_link, result_queue))

            thread3.start()
            thread4.start()

            # Add timeout to thread joins
            thread3.join(timeout=30)
            thread4.join(timeout=30)

            if thread3.is_alive() or thread4.is_alive():
                logger.error("Content processing threads timed out")
                raise TimeoutError("Content processing timed out")

            logger.debug("Completed custom_search_and_fetch_content successfully")
        except Exception as e:
            logger.exception(f"Error in custom_search_and_fetch_content: {str(e)}")
            result_queue.put(("data_title_Summary_str", f"获取搜索内容时出错: {str(e)}"))
            result_queue.put(("link_detail_string", ""))

    def Google_Search_run(self, question):
        try:
            logger.debug(f"Starting Google search for question: {question}")
            # get_google_result.set_global_proxy(self.proxy_url_global)

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
            # 创建并启线程
            thread1 = threading.Thread(target=self.get_google_answer, args=(question, results_queue))
            thread2 = threading.Thread(target=self.custom_search_and_fetch_content, args=(question, results_queue))

            logger.debug("Starting search threads")
            thread1.start()
            thread2.start()

            # Add timeout to thread joins
            thread1.join(timeout=30)
            thread2.join(timeout=30)

            if thread1.is_alive() or thread2.is_alive():
                logger.error("Search threads timed out")
                raise TimeoutError("Search operation timed out")

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
            当前关键字搜索的答案框据：
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

        except Exception as e:
            logger.exception(f"Error in Google_Search_run: {str(e)}")
            return f"搜索过程中生错误: {str(e)}。请检查网络连接或重试。"

    def echo(self, message, history, llm_options_checkbox_group, collection_name_select,
             temperature_num, print_speed_step, tool_checkbox_group,
             Embedding_Model_select,
             local_data_embedding_token_max, local_private_llm_api, local_private_llm_key,
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
                streaming=True,
            )
        else:
            self.llm = ChatOpenAI(
                temperature=self.temperature_num_global,
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                model=self.llm_name_global,
                streaming=True,
            )

        self.tools = []  # 重置工具列表
        # Check if 'wolfram-alpha' is in the selected tools
        if "wolfram-alpha" in tool_checkbox_group:
            wolfram = WolframAlphaAPIWrapper()
            wolfram_tool = Tool(
                name="Wolfram Alpha",
                func=wolfram.run,
                description="Useful for when you need to answer questions about Math, Science, Technology, Culture, Society and Everyday Life"
            )
            self.tools.append(wolfram_tool)

        if "arxiv" in tool_checkbox_group:
            arxiv = ArxivAPIWrapper()
            arxiv_tool = Tool(
                name="Arxiv",
                func=arxiv.run,
                description="Useful for when you need to get information about scientific papers from arxiv.org. Input should be a search query."
            )
            self.tools.append(arxiv_tool)

        self.Google_Search_tool = Tool(
            name="Google_Search",
            func=self.Google_Search_run,
            description="""
                这是一个如果本地知识库无答案或问题需要网络搜索的Google搜索工具。
                1.你先根据我的问题提取出最适合Google搜索引擎搜索的关键字进行搜索,可以选择英语或者中文搜索
                2.同时增加一些搜索提示词包括(引号，时间，关键字)
                3.如果问题比较复杂，你可以一步一步的思考去搜索和回答
                4.确保每个回答都不仅基于数据，输出的回答必须包含深入、完整，充分反映你对问题的全面理解。
            """
        )
        self.Local_Search_tool = Tool(
            name="Local_Search",
            func=self.ask_local_vector_db,
            description="""
                这是一个本地知识库搜索工具，你可以优使用本地搜索并总结回答。
                1.你先根我的问题提取出最适合embedding模型向量匹配的关键字进行搜索。
                2.注意你需要提出非常有针对性准确的问题和回答。
                3.如果问题比较复杂，可以将复杂的问题进行行拆分，你可以一步一步的思考。
                4.确保每个回答都不仅基于数据，输出的回答必须包含深入、完整，充分反映你对问题的全面理解。
            """
        )

        self.Create_Image_tool = Tool(
            name="Create_Image",
            func=self.createImageByBing,
            description="""
                这是一个图片生成工具，当我的问题中明确需要画图，你就可以使用该工具并生成图片
                1。当你回关于需要使用bing来生成什么画图、照片时很有用，先提取生成图片的提示词，然后调用该工具。
                2.并严格按照Markdown语法: [![图片描述](图片链接)](图片链接)。
                3.如果生成的图片链接数量大于1，将其全部严格按照Markdown语法: [![图片描述](图片链接)](图片链接)。
                4.如果问题比较复杂，可以将复杂的问题进行拆分，你可以一步一步的思考。
                """
        )

        self.tools = [self.math_tool]  # 确保始终包含 Calculator 工具
        # Initialize flags for additional tools
        flag_get_Local_Search_tool = False
        # Check for additional tools and append them if not already in the list
        for tg in tool_checkbox_group:
            if tg == "Google Search" and self.Google_Search_tool not in self.tools:
                response = "Google Search 工具加入 回答中..........."
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[:i + int(print_speed_step)]
                self.tools.append(self.Google_Search_tool)

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
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-mpnet-base-v2",
                        cache_folder="models"
                    )
                    self.Embedding_Model_select_global = 1

                flag_get_Local_Search_tool = True

            elif tg == "Create Image" and self.Create_Image_tool not in self.tools:
                response = "Create Image 工具加入 回答中..........."
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[:i + int(print_speed_step)]
                self.tools.append(self.Create_Image_tool)

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

        if llm_Agent_checkbox_group == "openai-functions":
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            llm_with_tools = self.llm.bind(
                functions=[format_tool_to_openai_function(t) for t in self.tools]
            )

            agent = (
                    {
                        "input": lambda x: x["input"],
                        "agent_scratchpad": lambda x: format_to_openai_function_messages(
                            x["intermediate_steps"]
                        ),
                    }
                    | prompt
                    | llm_with_tools
                    | OpenAIFunctionsAgentOutputParser()
            )

            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
            )

            try:
                response = ""
                for chunk in agent_executor.stream({"input": message}):
                    if "output" in chunk:
                        response += chunk["output"]
                        yield response
                    elif "intermediate_step" in chunk:
                        self.intermediate_steps_log = str(chunk["intermediate_step"])

            except Exception as e:
                yield f"发生错误：{str(e)}"
                logger.error(f"Error in agent execution: {str(e)}")
        elif llm_Agent_checkbox_group == "ZeroShotAgent-memory":
            # 修改 prefix 和 suffix 以更好地处理对话
            prefix = """你是一个智能AI助手，擅长通过逻辑思考来解决问题。在回答问题时，请遵循以下思考步骤：

1. 首先，仔细分析用户的问题，理解问题的核心需求
2. 思考是否可以直接回答，还是需要使用工具来获取更多信息
3. 如果问题复杂，可以将其分解成多个子问题逐步解决
4. 在使用工具时，要明确说明使用原因和预期结果

请严格按照以下格式回复：

Thought: 分析问题并说明思考过程
(可选) Action: 如果需要使用工具，选择合适的工具
(可选) Action Input: 输入到工具的具体内容
(可选) Observation: 工具返回的结果
... (如果需要，可以重复上述思考-行动-观察循环)
Thought: 总结所有信息，形成最终答案
Final Answer: 给出完整、准确、有条理的回答

当前可用的工具有:"""

            suffix = """请记住：
1. 优先通过自己的知识和逻辑思考来回答
2. 只在确实需要时才使用工具
3. 回答要有条理、完整且符合逻辑
4. 如果不确定，要诚实说明并给出最佳建议

历史对话:
{chat_history}

当前问题: {input}

思考过程:
{agent_scratchpad}

让我们一步一步地思考这个问题..."""

            prompt = ZeroShotAgent.create_prompt(
                self.tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )

            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools, verbose=True)

            # 创建一个回调处理器来捕获中间步骤
            class VerboseHandler(BaseCallbackHandler):
                def __init__(self):
                    self.steps = []
                    self.current_iteration = 0  # 添加轮次计数器
                    super().__init__()
                
                def on_agent_action(self, action, color=None, **kwargs):
                    try:
                        # 确保思考过程被正确记录
                        if hasattr(action, 'log') and action.log:
                            self.current_iteration += 1  # 增加轮次计数
                            self.steps.append(f"\n**第 {self.current_iteration} 轮思考过程**")
                            self.steps.append(f"**思考:** {action.log}")
                        
                        # 确保工具名称被正确记录
                        if hasattr(action, 'tool'):
                            self.steps.append(f"**行动:** {action.tool}")
                        
                        # 确保工具输入被正确记录
                        if hasattr(action, 'tool_input'):
                            self.steps.append(f"**输入:** {action.tool_input}")
                    except Exception as e:
                        self.steps.append(f"**注意:** 行动记录出现问题: {str(e)}")
                    
                def on_agent_observation(self, observation, color=None, **kwargs):
                    try:
                        if observation:
                            self.steps.append(f"**观察:** {observation}")
                    except Exception as e:
                        self.steps.append(f"**注意:** 观察记录出现问题: {str(e)}")
                    
                def on_agent_finish(self, finish, color=None, **kwargs):
                    try:
                        if hasattr(finish, 'log') and finish.log:
                            self.steps.append(f"\n**最终思考**")
                            self.steps.append(f"**思考:** {finish.log}")
                        
                        if hasattr(finish, 'return_values'):
                            if isinstance(finish.return_values, dict) and "output" in finish.return_values:
                                self.steps.append(f"**最终答案:** {finish.return_values['output']}")
                            else:
                                self.steps.append(f"**最终答案:** {str(finish.return_values)}")
                    except Exception as e:
                        self.steps.append(f"**注意:** 完成记录出现问题: {str(e)}")

            handler = VerboseHandler()
            
            # 修改 agent_chain 配置
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                verbose=True,
                memory=self.memory,
                max_iterations=3,
                handle_parsing_errors=True,
                early_stopping_method="generate",
                callbacks=[handler],
                return_intermediate_steps=True  # 添加这个参数以确保获取中间步骤
            )

            try:
                # 获取聊天历史
                chat_history = self.memory.load_memory_variables({})["chat_history"]

                # 运行agent_chain
                result = agent_chain(
                    {"input": message, "chat_history": chat_history},
                    include_run_info=True
                )

                # 组合所有步骤并输出
                if handler.steps:
                    full_output = "\n".join(handler.steps)
                else:
                    # 如果没有捕获到步骤，尝试从结果中提取
                    steps = []
                    if "intermediate_steps" in result:
                        for step in result["intermediate_steps"]:
                            if len(step) >= 2:
                                action, observation = step
                                steps.append(f"**思考:** {action.log if hasattr(action, 'log') else ''}")
                                steps.append(f"**行动:** {action.tool if hasattr(action, 'tool') else ''}")
                                steps.append(f"**输入:** {action.tool_input if hasattr(action, 'tool_input') else ''}")
                                steps.append(f"**观察:** {observation}")
                    
                    if "output" in result:
                        steps.append(f"**最终答案:** {result['output']}")
                    
                    full_output = "\n".join(steps)

                yield full_output

            except Exception as e:
                error_msg = f"Agent执行过程中发生错误：{str(e)}"
                logger.error(error_msg)
                yield error_msg

    def update_collection_name(self):
        # 获取已存在的collection的名称列表
        collections = self.client.list_collections()
        collection_name_choices = [collection.name for collection in collections]
        # 返回新的下拉列表组件
        return gr.Dropdown(
            choices=collection_name_choices,
            value=collection_name_choices[0] if collection_name_choices else None
        )

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as self.interface:
            with gr.Row(equal_height=True):  # 设置行等高
                with gr.Column(scale=3):
                    # 左侧列: 所控件
                    with gr.Row():
                        with gr.Group():
                            gr.Markdown("### Language Model Selection")
                            llm_options = ["gpt-4o", "Private-LLM-Model"]
                            llm_options_checkbox_group = gr.Dropdown(llm_options, label="LLM Model Select Options",
                                                                     value=llm_options[0])
                            gr.Markdown(
                                "Note: When selecting a Private LLM Model, ensure that you consider the Private LLM Settings.")

                            llm_Agent = ["openai-functions", "ZeroShotAgent-memory"]
                            llm_Agent_checkbox_group = gr.Dropdown(llm_Agent, label="LLM Agent Type Options",
                                                                   value=llm_Agent[0])

                        with gr.Group():
                            gr.Markdown("### Private LLM Settings")
                            local_private_llm_name = gr.Textbox(value="gpt-4o-mini", label="Private llm name")
                            local_private_llm_api = gr.Textbox(value="https://api.chatanywhere.tech",
                                                               label="Private llm openai-api base")
                            local_private_llm_key = gr.Textbox(value="EMPTY", label="Private llm openai-api key")

                    with gr.Row():
                        with gr.Group():
                            gr.Markdown("### Additional Tools")

                            tool_options = ["Google Search", "Local Knowledge Search", "wolfram-alpha", "arxiv",
                                            "Create Image"]
                            tool_checkbox_group = gr.CheckboxGroup(tool_options, label="Tools Select")
                            gr.Markdown("Note: Select the tools you want to use.")

                        with gr.Group():
                            gr.Markdown("### Knowledge Collection Settings")
                            collection_name_select = gr.Dropdown(
                                choices=[],  # 初始为空列表
                                label="Select existed Collection",
                                value=None  # 初始值为 None
                            )
                            Refresh_button = gr.Button("Refresh Collection", variant="secondary")
                            Refresh_button.click(fn=self.update_collection_name, outputs=collection_name_select)

                            temperature_num = gr.Slider(0, 1, label="Temperature")
                            print_speed_step = gr.Slider(5, 10, label="Print Speed Step", step=1)

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

                    chatbot = gr.ChatInterface(
                        self.echo,
                        additional_inputs=[llm_options_checkbox_group, collection_name_select,
                                           temperature_num, print_speed_step, tool_checkbox_group,
                                           Embedding_Model_select,
                                           local_data_embedding_token_max, local_private_llm_api,
                                           local_private_llm_key,
                                           local_private_llm_name, llm_Agent_checkbox_group],
                        title="RainbowGPT-Agent",
                        css=custom_css,
                        description="""
                            <div class='footer-email'>
                                <p>How to reach us：<a href='mailto:zhujiadongvip@163.com'>zhujiadongvip@163.com</a></p>
                            </div>
                        """,
                        theme="soft",
                        fill_height=True,
                        autoscroll=True,
                        type='messages'
                    )

    def launch(self):
        return self.interface
