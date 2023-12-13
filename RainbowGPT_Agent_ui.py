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
from Rainbow_utils.get_gradio_theme import Seafoam
from Rainbow_utils.get_tokens_cal_filter import filter_chinese_english_punctuation, num_tokens_from_string, \
    truncate_string_to_max_tokens
from Rainbow_utils import get_google_result
from Rainbow_utils import get_prompt_templates

load_dotenv()

seafoam = Seafoam()

script_name = os.path.basename(__file__)
logfile = script_name + ".log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)

persist_directory = ".chromadb/"
client = chromadb.PersistentClient(path=persist_directory)
collection_name_select_global = None

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
# 文档切分的长度
input_chunk_size_global = None
# 本地知识库嵌入token max
local_data_embedding_token_max_global = None
# 在文件顶部定义docsearch_db
docsearch_db = None
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


def ask_local_vector_db(question):
    global docsearch_db
    global llm
    global embeddings
    global Embedding_Model_select_global
    global temperature_num_global
    global llm_name_global
    global input_chunk_size_global
    global local_data_embedding_token_max_global
    global human_input_global
    global tools
    global collection_name_select_global
    global local_private_llm_key_global
    global local_private_llm_api_global
    global local_private_llm_name_global

    if Embedding_Model_select_global == 0:
        embeddings = OpenAIEmbeddings()
        embeddings.show_progress_bar = True
        embeddings.request_timeout = 20
    elif Embedding_Model_select_global == 1:
        embeddings = HuggingFaceEmbeddings()

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

    local_search_prompt = PromptTemplate(
        input_variables=["combined_text", "human_input", "human_input_first"],
        template=local_search_template,
    )
    # 本地知识库工具
    local_chain = LLMChain(
        llm=llm, prompt=local_search_prompt,
        verbose=True,
        return_final_only=True,  # 指示是否仅返回最终解析的结果
    )

    docs = []
    if Embedding_Model_select_global == 0:
        print("OpenAIEmbeddings Search")
        # 结合基础检索器+Embedding上下文压缩
        # 将稀疏检索器（如 BM25）与密集检索器（如嵌入相似性）相结合
        chroma_retriever = docsearch_db.as_retriever(search_kwargs={"k": 30})

        # 将压缩器和文档转换器串在一起
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                               base_retriever=chroma_retriever)
        # compressed_docs = compression_retriever.get_relevant_documents(question, tools=tools)

        the_collection = client.get_collection(name=collection_name_select_global)
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
    elif Embedding_Model_select_global == 1:
        print("HuggingFaceEmbedding Search")
        chroma_retriever = docsearch_db.as_retriever(search_kwargs={"k": 30})
        # docs = chroma_retriever.get_relevant_documents(question)
        # chroma_vectorstore = Chroma.from_texts(the_doc_llist, embeddings)
        # chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 10})

        the_collection = client.get_collection(name=collection_name_select_global)
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
        if total_toknes + tokens <= (int(local_data_embedding_token_max_global)):
            cleaned_matches.append(cleaned_context)
            total_toknes += tokens
        else:
            last_index = index
            break
    print("Embedding了 ", str(last_index + 1), " 个知识库文档块")
    # 将清理过的匹配项组合合成一个字符串
    combined_text = " ".join(cleaned_matches)

    answer = local_chain.predict(combined_text=combined_text, human_input=question,
                                 human_input_first=human_input_global)
    return answer


Local_Search_tool = Tool(
    name="Local_Search",
    func=ask_local_vector_db,
    description="""
        这是一个本地知识库搜索工具，你可以优先使用本地搜索并总结回答。
        1.你先根据我的问题提取出最适合embedding模型向量匹配的关键字进行搜索。
        2.注意你需要提出非常有针对性准确的问题和回答。
        3.如果问题比较复杂，可以将复杂的问题进行拆分，你可以一步一步的思考。
        """
)


def get_google_answer(question, result_queue):
    google_answer_box = get_google_result.selenium_google_answer_box(
        question, "Stock_Agent/chromedriver-120.0.6099.56.0.exe")
    # 使用正则表达式保留中文、英文和标点符号
    google_answer_box = filter_chinese_english_punctuation(google_answer_box)
    result_queue.put(("google_answer_box", google_answer_box))


def process_data_title_summary(data_title_Summary, result_queue):
    data_title_Summary_str = ''.join(data_title_Summary)
    result_queue.put(("data_title_Summary_str", data_title_Summary_str))


def process_custom_search_link(custom_search_link, result_queue):
    link_detail_res = []
    for link in custom_search_link[:1]:
        website_content = get_google_result.get_website_content(link)
        if website_content:
            link_detail_res.append(website_content)

    link_detail_string = '\n'.join(link_detail_res)
    link_detail_string = filter_chinese_english_punctuation(link_detail_string)
    result_queue.put(("link_detail_string", link_detail_string))


def custom_search_and_fetch_content(question, result_queue):
    custom_search_link, data_title_Summary = get_google_result.google_custom_search(question)

    # 创建新的线程来处理 data_title_Summary 和 custom_search_link
    thread3 = threading.Thread(target=process_data_title_summary, args=(data_title_Summary, result_queue))
    thread4 = threading.Thread(target=process_custom_search_link, args=(custom_search_link, result_queue))

    thread3.start()
    thread4.start()
    thread3.join()
    thread4.join()


def Google_Search_run(question):
    global llm
    global embeddings
    global Embedding_Model_select_global
    global temperature_num_global
    global llm_name_global
    global human_input_global
    global tools
    global proxy_url_global
    global local_data_embedding_token_max_global
    global local_private_llm_key_global
    global local_private_llm_api_global
    global local_private_llm_name_global

    get_google_result.set_global_proxy(proxy_url_global)

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

    local_search_prompt = PromptTemplate(
        input_variables=["combined_text", "human_input", "human_input_first"],
        template=google_search_template,
    )
    # 本地知识库工具
    local_chain = LLMChain(
        llm=llm, prompt=local_search_prompt,
        verbose=True,
        return_final_only=True,  # 指示是否仅返回最终解析的结果
    )

    # 创建一个队列来存储线程结果
    results_queue = queue.Queue()

    # 创建并启动线程
    thread1 = threading.Thread(target=get_google_answer, args=(question, results_queue))
    thread2 = threading.Thread(target=custom_search_and_fetch_content, args=(question, results_queue))

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
                                                   local_data_embedding_token_max_global,
                                                   "cl100k_base",
                                                   step_size=256)

    answer = local_chain.predict(combined_text=truncated_text, human_input=question,
                                 human_input_first=human_input_global)

    return answer


Google_Search_tool = Tool(
    name="Google_Search",
    func=Google_Search_run,
    description="""
    如果本地知识库中无答案或问题需要网络搜索可用这个互联网搜索工具进行搜索问答。
    1.你先根据我的问题提取出最适合Google搜索引擎搜索的关键字进行搜索,可以选择英语或者中文搜索
    2.同时增加一些搜索提示词包括(使用引号，时间范围，关键字和符号)
    3.如果问题比较复杂，你可以一步一步的思考去搜索和回答
    """
)


def echo(message, history, llm_options_checkbox_group, collection_name_select, collection_checkbox_group,
         new_collection_name,
         temperature_num, print_speed_step, tool_checkbox_group, uploaded_files, Embedding_Model_select,
         input_chunk_size, local_data_embedding_token_max, Google_proxy,
         local_private_llm_api,
         local_private_llm_key,
         local_private_llm_name):
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

    local_private_llm_name_global = str(local_private_llm_name)
    local_private_llm_api_global = str(local_private_llm_api)
    local_private_llm_key_global = str(local_private_llm_key)

    collection_name_select_global = str(collection_name_select)

    human_input_global = message
    local_data_embedding_token_max_global = int(local_data_embedding_token_max)
    input_chunk_size_global = int(input_chunk_size)
    temperature_num_global = float(temperature_num)
    llm_name_global = str(llm_options_checkbox_group)

    if Embedding_Model_select == "Openai Embedding" or Embedding_Model_select == "" or Embedding_Model_select == None:
        embeddings = OpenAIEmbeddings()
        embeddings.show_progress_bar = True
        embeddings.request_timeout = 20
        Embedding_Model_select_global = 0
    elif Embedding_Model_select == "HuggingFace Embedding":
        embeddings = HuggingFaceEmbeddings(cache_folder="models")
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

    tools = []  # 重置工具列表
    tools = load_tools(["llm-math", "arxiv"], llm=llm)

    flag_get_Local_Search_tool = False
    for tg in tool_checkbox_group:
        if tg == "Google Search" and Google_Search_tool not in tools:
            tools.append(Google_Search_tool)
            response = "Google Search 工具加入 回答中..........."

            # 设置代理（替换为你的代理地址和端口）
            proxy_url_global = str(Google_proxy)

            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]
        elif tg == "Local Knowledge Search" and Local_Search_tool not in tools:
            tools.append(Local_Search_tool)
            response = "Local Knowledge Search 工具加入 回答中..........."
            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]
            if Local_Search_tool in tools:
                flag_get_Local_Search_tool = True

    if message == "" and (
            (collection_checkbox_group == "Read Existing Collection") or (collection_checkbox_group == None)
            or collection_checkbox_group == "None"):
        response = "哎呀！好像有点小尴尬，您似乎忘记提出问题了。别着急，随时输入您的问题，我将尽力为您提供帮助！"
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]
        return

    if collection_checkbox_group == "Create New Collection":
        if new_collection_name == None or new_collection_name == "":
            response = "新知识库的名字没有写，创建中止！"
            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]
            return

        # 获取当前脚本所在文件夹的绝对路径
        current_script_folder = os.path.abspath(os.path.dirname(__file__))
        # 获取当前时间并格式化为字符串
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        base_folder = "\\data\\" + str(new_collection_name)
        # 根据时间创建唯一的文件夹名
        save_folder = current_script_folder + f"{base_folder}_{current_time}"

        try:
            os.makedirs(save_folder, exist_ok=True)
        except Exception as e:
            print(f"创建文件夹失败：{e}")

        # 保存每个文件到指定文件夹
        try:
            for file in uploaded_files:
                # 将文件指针重置到文件的开头
                source_file_path = str(file.orig_name)
                # 读取文件内容
                with open(source_file_path, 'rb') as source_file:
                    file_data = source_file.read()
                # 使用原始文件名构建保存文件的路径
                save_path = os.path.join(save_folder, os.path.basename(file.orig_name))
                # 保存文件
                # 保存文件到目标文件夹
                with open(save_path, 'wb') as target_file:
                    target_file.write(file_data)
        except Exception as e:
            print(f"保存文件时发生异常：{e}")

        # 设置向量存储相关配置
        response = "开始转换文件夹中的所有数据成知识库........"
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]

        loader = DirectoryLoader(str(save_folder), show_progress=True,
                                 use_multithreading=True,
                                 silent_errors=True)

        documents = loader.load()
        if documents == None:
            response = "文件读取失败！" + str(save_folder)
            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]
            return

        response = str(documents)
        for i in range(0, len(response), len(response) // 3):
            yield response[: i + (len(response) // 3)]

        print("documents len= ", documents.__len__())
        response = "文档数据长度为： " + str(documents.__len__())
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]

        intput_chunk_overlap = 24
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=int(input_chunk_size),
                                              chunk_overlap=int(intput_chunk_overlap))

        texts = text_splitter.split_documents(documents)
        print(texts)
        response = str(texts)
        for i in range(0, len(response), len(response) // 3):
            yield response[: i + (len(response) // 3)]
        print("after split documents len= ", texts.__len__())
        response = "切分之后文档数据长度为：" + str(texts.__len__()) + " 数据开始写入词向量库....."
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]

        # Collection does not exist, create it
        docsearch_db = Chroma.from_documents(documents=texts, embedding=embeddings,
                                             collection_name=str(new_collection_name + "_" + current_time),
                                             persist_directory=persist_directory,
                                             Embedding_Model_select=Embedding_Model_select_global)

        response = "知识库建立完毕！请去打开读取知识库按钮并输入宁的问题！"
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]

        return

    if flag_get_Local_Search_tool:
        if collection_checkbox_group == "Read Existing Collection":
            # Collection exists, load it
            if collection_name_select:
                print(f"{collection_name_select}", " Collection exists, load it")
                response = f"{collection_name_select}" + "知识库已经创建, 正在加载中...请耐心等待我的回答...."
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[: i + int(print_speed_step)]
                docsearch_db = Chroma(client=client, embedding_function=embeddings,
                                      collection_name=collection_name_select)

            else:
                response = "没有选中任何知识库，请至少选择一个知识库，回答中止！"
                for i in range(0, len(response), int(print_speed_step)):
                    yield response[: i + int(print_speed_step)]
                return
        elif collection_checkbox_group == None:
            response = "打开知识库搜索工具但是没有打开读取知识库开关！"
            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]
            return
    if not flag_get_Local_Search_tool and collection_checkbox_group == "Read Existing Collection":
        response = "读取知识库但是没有打开知识库搜索工具！失去本地知识库搜索能力！使用模型本身记忆或者其他工具回答！"
        for i in range(0, len(response), int(print_speed_step)):
            yield response[: i + int(print_speed_step)]

    # 初始化agent代理
    agent_open_functions = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
        max_iterations=10,
        early_stopping_method="generate",
        handle_parsing_errors=True,  # 初始化代理并处理解析错误
        # handle_parsing_errors="Check your output and make sure it conforms!",
        callbacks=[handler],
        # return_intermediate_steps=True,
    )

    try:
        response = agent_open_functions.run(message)
    except Exception as e:
        response = f"发生错误：{str(e)}"
    for i in range(0, len(response), int(print_speed_step)):
        yield response[: i + int(print_speed_step)]
    # response = agent_open_functions.run(message)
    # return response


# 定义一个函数，根据Local Knowledge Collection Select Options的值来返回Select existed Collection的选项
def update_collection_name(collection_option):
    if collection_option == "None":
        collection_name_choices = ["..."]
    elif collection_option == "Read Existing Collection":
        # 获取已存在的collection的名称列表
        collections = client.list_collections()
        collection_name_choices = []
        for collection in collections:
            collection_name = collection.name
            collection_name_choices.append(collection_name)
    elif collection_option == "Create New Collection":
        collection_name_choices = ["..."]
    # 调用gr.Dropdown.update方法，传入新的选项列表
    return gr.Dropdown.update(choices=collection_name_choices)


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
                    local_private_llm_name = gr.Textbox(value="Qwen-72B-Chat", label="Private llm name")

                with gr.Group():
                    gr.Markdown("### Private LLM Settings")
                    local_private_llm_api = gr.Textbox(value="http://172.16.0.160:8000/v1",
                                                       label="Private llm openai-api base")
                    local_private_llm_key = gr.Textbox(value="EMPTY", label="Private llm openai-api key")

            with gr.Row():
                with gr.Group():
                    gr.Markdown("### Additional Tools")
                    Google_proxy = gr.Textbox(value="http://localhost:7890", label="System Http Proxy")
                    tool_options = ["Google Search", "Local Knowledge Search"]
                    tool_checkbox_group = gr.CheckboxGroup(tool_options, label="Tools Select Options")

                    temperature_num = gr.Slider(0, 1, label="Temperature")
                    print_speed_step = gr.Slider(10, 20, label="Print Speed Step", step=1)

                with gr.Group():
                    gr.Markdown("### Knowledge Collection Settings")
                    collection_options = ["None", "Read Existing Collection", "Create New Collection"]
                    collection_checkbox_group = gr.Radio(collection_options,
                                                         label="Local Knowledge Collection Select Options",
                                                         value=collection_options[0])
                    collection_name_select = gr.Dropdown(["..."], label="Select existed Collection", value="...")
                    collection_checkbox_group.change(fn=update_collection_name, inputs=collection_checkbox_group,
                                                     outputs=collection_name_select)
                    input_chunk_size = gr.Textbox(value="512", label="Create Chunk Size")

            with gr.Row():
                with gr.Group():
                    gr.Markdown("### Embedding Data Settings")
                    Embedding_Model_select = gr.Radio(["Openai Embedding", "HuggingFace Embedding"],
                                                      label="Embedding Model Select Options", value="Openai Embedding")
                    local_data_embedding_token_max = gr.Slider(1024, 12288, step=2, label="Embeddings Data Max Tokens",
                                                               value=2048)

            with gr.Row():
                with gr.Group():
                    gr.Markdown("### Create Collection Settings")
                    new_collection_name = gr.Textbox("", label="New Collection Name")
                    uploaded_files = gr.File(file_count="multiple", label="Upload Files")

        with gr.Column(scale=5):
            # 右侧列: Chat Interface
            gr.ChatInterface(
                echo, additional_inputs=[llm_options_checkbox_group, collection_name_select, collection_checkbox_group,
                                         new_collection_name,
                                         temperature_num, print_speed_step, tool_checkbox_group, uploaded_files,
                                         Embedding_Model_select, input_chunk_size, local_data_embedding_token_max,
                                         Google_proxy, local_private_llm_api, local_private_llm_key,
                                         local_private_llm_name],
                title="""
    <h1 style='text-align: center; margin-bottom: 1rem; font-family: "Courier New", monospace;
               background: linear-gradient(135deg, #9400D3, #4B0082, #0000FF, #008000, #FFFF00, #FF7F00, #FF0000);
               -webkit-background-clip: text;
               color: transparent;'>
        RainbowGPT-Agent
    </h1>
    """,
                description="<div style='font-size: 14px; ...'>How to reach me: <a href='mailto:zhujiadongvip@163.com'>zhujiadongvip@163.com</a></div>",
                # css=".gradio-container {background-color: #f0f0f0;}"  # Add your desired background color here
            )

RainbowGPT.queue().launch()
# RainbowGPT.queue().launch(share=True)
