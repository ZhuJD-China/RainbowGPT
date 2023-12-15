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

    def createImageByBing(self, input):
        auth_cooker = "_IDET=MIExp=0; ipv6=hit=1702373308373&t=4; MUID=10648260268A616E003191AF27C56003; MUIDB=10648260268A616E003191AF27C56003; _EDGE_V=1; SRCHD=AF=NOFORM; SRCHUID=V=2&GUID=C5B39C165E044DA2B4BF9239D9C2A632&dmnchg=1; _UR=cdxcls=0&QS=0&TQS=0; MicrosoftApplicationsTelemetryDeviceId=ba566366-220c-4659-a5cb-071d80e7a999; PPLState=1; SnrOvr=X=rebateson; MMCASM=ID=CB2ED8CDE4594CE79337F56F567CA9BF; MSCCSC=1; BCP=AD=1&AL=1&SM=1; ANON=A=DD1208FB49DD920898E3E968FFFFFFFF&E=1d23&W=1; NAP=V=1.9&E=1cc9&C=h9v0nY_oxh2iCzz9WRmsSUoQmojpK7s1FeCqska8xP_wTIRZBeeSvg&W=1; EDGSRCHHPGUSR=CIBV=1.1381.6&udstone=Creative&udstoneopts=h3imaginative,gencontentv3,fluxv1,flxegctxv3,egctxcplt,fluxv14l; _Rwho=u=d; SRCHS=PC=U531; USRLOC=HS=1&ELOC=LAT=30.23148536682129|LON=119.70948791503906|N=%E4%B8%B4%E5%AE%89%E5%8C%BA%EF%BC%8C%E6%B5%99%E6%B1%9F%E7%9C%81|ELT=4|&CLOC=LAT=30.2323|LON=119.7174|A=8116|TS=231211200954|SRC=I; ipv6=hit=1702365893520&t=4; _HPVN=CS=eyJQbiI6eyJDbiI6MiwiU3QiOjAsIlFzIjowLCJQcm9kIjoiUCJ9LCJTYyI6eyJDbiI6MiwiU3QiOjAsIlFzIjowLCJQcm9kIjoiSCJ9LCJReiI6eyJDbiI6MiwiU3QiOjAsIlFzIjowLCJQcm9kIjoiVCJ9LCJBcCI6dHJ1ZSwiTXV0ZSI6dHJ1ZSwiTGFkIjoiMjAyMy0xMi0xMlQwMDowMDowMFoiLCJJb3RkIjowLCJHd2IiOjAsIlRucyI6MCwiRGZ0IjpudWxsLCJNdnMiOjAsIkZsdCI6MCwiSW1wIjo4LCJUb2JicyI6MH0=; SRCHHPGUSR=CW=723&CH=966&SCW=708&SCH=966&BRW=NOTP&BRH=M&SRCHLANG=zh-Hans&PV=10.0.0&HV=1702362299&BZA=0&PRVCW=1278&PRVCH=966&DPR=1.0&UTC=480&DM=0&EXLTT=1&WTS=63837959089&IG=CE2CB999A8EF40C2A6C8199931FD89CD; ai_session=9TXXKexi1yLp0N4E7jqDuP|1702369684940|1702369684940; CSRFCookie=62d1da8c-1afd-435a-9128-001034b3a8c6; _EDGE_S=SID=0D913A7E2C056D64305B299B2D956C8D; KievRPSSecAuth=FABCBBRaTOJILtFsMkpLVWSG6AN6C/svRwNmAAAEgAAACJKHqoGXhERjAATjE4qzwElh7BczRIO6RQc+0of7ICFWbKn244NEl3HfOXYoxL8eLOARdD0i/Bbc6qHtuUpWu8x3sEr6unCt1HbHdMWSVzHknf/DQIpk3dcoJkMtYBh4gMnI0Fqfi3+2nKw5t4SdOOrkoF0ENKaEGNmAHKKkHoSTgJ9hglrmvO5ou3ltJcLL7mS4W06tYpsj8iKiyWpwdqn0D6y2KeY/goT1ibU4PjqBJ2C2Sm2Q8S0osSE7uaLesPdUaA9sHgOR/vG2Ucs2kxz9NuDESrR3EvW3ZgAWhbsdBA3ZbZKrtTVXdOhliGDD8jrgthz85m/DIAtFXQ2qQWAWjVUVWwxv/HiYnsmO3q8hTkCOXUTaRne3gLWnIFWDaMeub6xy0nXaiF9EQx4IlaudAel1UZHHHsc1bkzv/PaKSzHV+G2pkQS3ii+K51q/D2S6LfKaknX0kliXg2UEg60OoP3h35T0Ut2B3JqYQ1HQ0KU0WVmgru+IqiwEOF/YylL72H+5en1VNcFwgNoLq3h5W92b1G4yTPRga/IPXKWyb1B0UwTHDYN9j71oDUTKbJsQ/vnCUVKRhD9gcj2bmIeIauczLuj53Kl371ntH1DhzLYL45oNdKojJeI9lo/fEWiYthDXT/OWESbaxpM84KXngcqJ7Hwfis2m/oj7OlJ1Y66/sOorweX0Ny1aiKAwFdViEvnKz6nHZdshQQPEfE1iId9Cbi7VE4rH1oQsnOAHJV3bVxb2vUCNhdaOkCZ7wFhGBca3rnNY1HT4D0+oyP/SzOArEi4UTdI3Efn7pE3X4lHWFKTwV3rapncvvJZJCwcIROYmHm1ACK/dvwqQRVVqIX4k/hkIPGIgZUpa/AqB/TcYYpHLCmccjbR+katCzCTzqcxxSSyvLAiCUBnEWHnYSWdVXyLnSrJ84TnXGOuzqtoVsecGHF6aqh/cD1lj3cUKfGg9v4rvi5wsuUZyNhRpz/kavM1Tyl+641GPJA4v0mlJGNd9TQZHWDctBzUmdE3jIkujoTkdb2BQM+ygpsvQ0LJEzZiK423mK5qcua0EldAc5iJ62b5tC8H14DWVBhwHOWuyjPegFSKIBIfo1HX/ZBbK4isavWTKHSySAttmwuiETq9MoyjdXrtdJ8qNS3Rp8VTA+DkpEeVI9c4ox4fliGBeKftcLhGL56dC9va0uAapjPWZF8T90ATzAxJtyH5rDQQ0c8UHmfk+RmGovS8HXqMFi4OmMjXBmFiQN9Bozv4I8YorerxkWOorCChfQMEnP7FTuQPoSKfx6+MpKbb90FdSSlIVdu5yETdXr8Bs90wSyVBvX94ung99Do5tdDVghZS22++U0BAbg+NpIkyPOLkHZCIEizKGFACyeQpnEIjbrSh4/1dTjioyZfAVmg==; _U=1pWIFlkH5E72bC9U4gr_k29r6Pw2OobSjItX9ll3jCdJk-uq9SyIZMSHCyMVnZFU9TcWp0VmZ_6O0WLoUpwrL3BfSpG5_b7ue9IDkyx-XHqzaOl5Kp8IyBA5DlzrwM88PKDBf2GEkBvlzu8x6snhfPZlVBvsTOupk94UfCLhB2T9hFFtGfWsV2VVHGDAySfySZoPGfTO1e0aMXsHBOcoR3g; WLS=C=1740f8db0673bb65&N=; WLID=06PcLWNVQnxXk1KzGUtZjN/ZnCRUznLvRx0NFOsWrpRlygAWXXbB9OoZEe35+NZvlEx+SI1kVn62XIMLb798+9dp779DmGo2qO8VfW0GaD8=; SRCHUSR=DOB=20231120&T=1702369709000; _SS=SID=0B43890E7BA4602936459AEB7A5E61ED&R=14&RB=14&GB=0&RG=0&RP=5&PC=U531; _RwBf=W=1&r=1&ilt=2&ihpd=2&ispd=0&rc=14&rb=14&gb=0&rg=0&pc=5&mtu=0&rbb=0.0&g=0&cid=&clo=0&v=3&l=2023-12-12T08:00:00.0000000Z&lft=0001-01-01T00:00:00.0000000&aof=0&o=0&p=BINGCOPILOTWAITLIST&c=MY00IA&t=6891&s=2023-03-23T07:14:38.2733307+00:00&ts=2023-12-12T08:28:58.6778390+00:00&rwred=0&wls=2&wlb=0&lka=0&lkt=0&aad=0&TH=&mta=0&e=80DuGMy-G35l2r6pbllDzkQJiABuT83p7QRZ8vc7z5MMo1KP_tpT5zAA9mn03K_XOyrzCZcsFCgQ100DyipCBQ&A=DD1208FB49DD920898E3E968FFFFFFFF&ccp=0&wle=0; GC=T13aSA2CLeKvdQdLL_7H7G22sxNpmC1zm-nzVUoXW78a6C8vHD_fLdp6YmC4sAHY0QGDjywEY6r1U0s-F3LnPw; GI_FRE_COOKIE=gi_prompt=4&gi_fre=1&gi_sc=3"
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
        google_answer_box = get_google_result.selenium_google_answer_box(
            question, "Rainbow_utils/chromedriver.exe")
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
                这个一个如果本地知识库中无答案或问题需要网络搜索的Google网络搜索工具。
                1.你先根据我的问题提取出最适合Google搜索引擎搜索的关键字进行搜索,可以选择英语或者中文搜索
                2.同时增加一些搜索提示词包括(使用引号，时间范围，关键字和符号)
                3.如果问题比较复杂，你可以一步一步的思考去搜索和回答
                4.确保每个回答都不仅基于数据，输出的回答必须包含深入、完整，充分反映你对问题的全面理解。
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
                4.确保每个回答都不仅基于数据，输出的回答必须包含深入、完整，充分反映你对问题的全面理解。
            """
        )

        self.Create_Image_tool = Tool(
            name="Create_Image",
            func=self.createImageByBing,
            description="""
                这是一个图片生成工具，你可以使用该工具并生成图片。
                1。当你回答关于需要使用bing来生成什么、画图、照片时时很有用，先提取生成图片的提示词，然后调用该工具。
                2.并严格按照Markdown语法: [![图片描述](图片链接)](图片链接)。
                3.如果生成的图片链接数量大于1，将其全部严格按照Markdown语法: [![图片描述](图片链接)](图片链接)。
                4.如果问题比较复杂，可以将复杂的问题进行拆分，你可以一步一步的思考。
                """
        )

        # 默认开启
        self.tools.append(self.Create_Image_tool)

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

        # 初始化agent代理
        agent_open_functions = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=self.llm_Agent_checkbox_group,
            verbose=True,
            agent_kwargs=self.agent_kwargs,
            memory=self.memory,
            max_iterations=5,
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
                            gr.Markdown("Note: When Select Private-LLM-Model,you should take Private LLM Settings.")

                            llm_Agent = ["chat-zero-shot-react-description", "openai-functions",
                                         "zero-shot-react-description"]
                            llm_Agent_checkbox_group = gr.Dropdown(llm_Agent, label="LLM Agent Type Options",
                                                                   value=llm_Agent[0])

                        with gr.Group():
                            gr.Markdown("### Private LLM Settings")
                            local_private_llm_name = gr.Textbox(value="Qwen-*B-Chat", label="Private llm name")
                            local_private_llm_api = gr.Textbox(value="http://172.16.0.160:8000/v1",
                                                               label="Private llm openai-api base")
                            local_private_llm_key = gr.Textbox(value="EMPTY", label="Private llm openai-api key")

                    with gr.Row():
                        with gr.Group():
                            gr.Markdown("### Additional Tools")

                            tool_options = ["Google Search", "Local Knowledge Search", "wolfram-alpha"]
                            tool_checkbox_group = gr.CheckboxGroup(tool_options, label="Tools Select")
                            gr.Markdown("Note: 'Create Image','arxiv' Tools are enabled by default.")

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
                                                      local_data_embedding_token_max, local_private_llm_api,
                                                      local_private_llm_key,
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
