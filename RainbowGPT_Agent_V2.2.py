import chromadb
import openai
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
import tiktoken
import get_google_result
import datetime
import time
import os
from dotenv import load_dotenv
import gradio as gr
from gradio_theme import Seafoam
from loguru import logger
# 导入 langchain 模块的相关内容
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.callbacks import FileCallbackHandler
from langchain.chains import LLMChain
from langchain.document_loaders import DirectoryLoader
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain.retrievers import (
    ContextualCompressionRetriever,
    BM25Retriever,
    EnsembleRetriever
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from zh_en_langid import filter_chinese_english_punctuation

load_dotenv()

# 加载互联网共享接口
# 免费授权API接口网站: https://api.chatanywhere.org/v1/oauth/free/github/render
# 转发Host1: https://api.chatanywhere.com.cn (国内中转，延时更低，推荐)
# 转发Host2: https://api.chatanywhere.cn (国外使用,国内需要全局代理)
# openai.api_base = "https://api.chatanywhere.com.cn/v1"
# 加载环境变量中的 OpenAI API 密钥
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
# 打印 API 密钥
print(openai.api_key)

seafoam = Seafoam()

logfile = "RainbowGPT_Agent_V2.2.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)

persist_directory = ".chromadb/"
client = chromadb.PersistentClient(path=persist_directory)
collection_name_select_global = None

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

# 公共前部分
common_text_before = """
你是一位卓越的AI问答和知识库内容分析专家，为了更好地发挥你的专业性。
请在进行分析问答时着重关注以下知识库内容：
"""

# 公共后部分
common_text_after = """
我现在要问的问题是：{human_input}

希望你能展现深入的思考和对知识库内容的精准理解，以提供最为专业和有价值的回答。
请确保回答内容既详细又清晰，充分利用你的专业知识为问题提供全面而准确的解答。
"""

# local Search Prompt模版
local_search_template = common_text_before + """
以下双引号内是所搜索到的知识库数据：
“{combined_text}”
""" + common_text_after

# google Search Prompt模版
google_search_template = common_text_before + """
答案框数据类型包括特色片段、知识卡和实时结果，请你仔细分析答案框内容与我的问题后再决定是否利用这个数据回答。
以下双引号内是所搜索到的知识库数据：
“{combined_text}”
""" + common_text_after

# 全局工具列表创建
tools = []


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


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

    if Embedding_Model_select_global == 0:
        embeddings = OpenAIEmbeddings()
        embeddings.show_progress_bar = True
        embeddings.request_timeout = 20
    elif Embedding_Model_select_global == 1:
        embeddings = HuggingFaceEmbeddings()

    llm = ChatOpenAI(temperature=temperature_num_global, model=llm_name_global)

    local_search_prompt = PromptTemplate(
        input_variables=["combined_text", "human_input"],
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

    answer = local_chain.predict(combined_text=combined_text, human_input=human_input_global)
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


def truncate_string_to_max_tokens(input_string, max_tokens, tokenizer_name, step_size=5):
    """
    Truncate the input string to a maximum number of tokens with caching and a step size.

    Parameters:
    - input_string: The input string to truncate.
    - max_tokens: The maximum number of tokens allowed.
    - tokenizer_name: The name of the tokenizer.
    - step_size: The step size for increasing truncation length.

    Returns:
    - Truncated string.
    """

    # 从步长开始，逐渐增加截断长度，确保不超过最大令牌数
    for truncation_length in range(1, len(input_string) + 1, step_size):
        truncated_string = input_string[:truncation_length]

        # 计算当前截断字符串的令牌数量（使用带缓存的计算函数）
        current_tokens = num_tokens_from_string(truncated_string, tokenizer_name)

        # 如果当前令牌数超过最大令牌数，返回上一个截断字符串
        if current_tokens > max_tokens:
            return input_string[:truncation_length - step_size]

    # 如果整个字符串都符合要求，返回原始字符串
    return input_string


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

    get_google_result.set_global_proxy(proxy_url_global)

    llm = ChatOpenAI(temperature=temperature_num_global, model=llm_name_global)

    local_search_prompt = PromptTemplate(
        input_variables=["combined_text", "human_input"],
        template=google_search_template,
    )
    # 本地知识库工具
    local_chain = LLMChain(
        llm=llm, prompt=local_search_prompt,
        verbose=True,
        return_final_only=True,  # 指示是否仅返回最终解析的结果
    )

    google_answer_box = (get_google_result.selenium_google_answer_box
                         (question, "Stock_Agent/chromedriver-120.0.6099.56.0.exe"))

    # 使用正则表达式保留中文、英文和标点符号
    google_answer_box = filter_chinese_english_punctuation(google_answer_box)

    # Google_Search = GoogleSearchAPIWrapper()
    # GoogleSearchAPI_data = Google_Search.run(question)
    custom_search_link, data_title_Summary = get_google_result.google_custom_search(question)

    # 使用 join 方法将字符串列表连接成一个字符串
    data_title_Summary_str = ''.join(data_title_Summary)

    link_datial_res = []
    for link in custom_search_link[:1]:
        website_content = get_google_result.get_website_content(link)
        if website_content:
            link_datial_res.append(website_content)
    # Concatenate the strings in the list into a single string
    link_datial_string = '\n'.join(link_datial_res)

    # 使用正则表达式保留中文、英文和标点符号
    link_datial_string = filter_chinese_english_punctuation(link_datial_string)

    finally_combined_text = f"""当前关键字搜索的答案框数据：
    {google_answer_box}
    
    搜索结果相似度TOP10的网站的标题和摘要数据：
    {data_title_Summary_str}
    
    搜索结果相似度TOP1的网站的详细内容数据:
    {link_datial_string}
    """

    truncated_text = truncate_string_to_max_tokens(finally_combined_text,
                                                   local_data_embedding_token_max_global,
                                                   "cl100k_base",
                                                   step_size=256)

    answer = local_chain.predict(combined_text=truncated_text, human_input=human_input_global)

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
         input_chunk_size, local_data_embedding_token_max, Google_proxy):
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

    collection_name_select_global = str(collection_name_select)

    # 设置代理（替换为你的代理地址和端口）
    proxy_url_global = str(Google_proxy)
    os.environ['http_proxy'] = proxy_url_global
    os.environ['https_proxy'] = proxy_url_global

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
        embeddings = HuggingFaceEmbeddings()
        Embedding_Model_select_global = 1

    tools = []  # 重置工具列表
    llm_math_tool = load_tools(["arxiv"], llm=ChatOpenAI(model="gpt-3.5-turbo-16k"))
    tools.append(llm_math_tool[0])

    flag_get_Local_Search_tool = False
    for tg in tool_checkbox_group:
        if tg == "Google Search" and Google_Search_tool not in tools:
            tools.append(Google_Search_tool)
            response = "Google Search 工具加入 回答中..........."
            for i in range(0, len(response), int(print_speed_step)):
                yield response[: i + int(print_speed_step)]
        elif tg == "Local Knowledge Base Search" and Local_Search_tool not in tools:
            tools.append(Local_Search_tool)
            response = "Local Knowledge Base Search 工具加入 回答中..........."
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

    llm = ChatOpenAI(temperature=temperature_num_global, model=llm_name_global)

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

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

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
        with gr.Column():
            # 创建一个包含选项的多选框组
            llm_options = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview",
                           "gpt-4-vision-preview", "gpt-4", "gpt-3.5-turbo-16k", "gpt-3.5-turbo"]
            llm_options_checkbox_group = gr.Dropdown(llm_options, label="LLM Model Select Options",
                                                     value=llm_options[0])
            Google_proxy = gr.Textbox(value="http://localhost:7890", label="Google Http Proxy")
            tool_options = ["Google Search", "Local Knowledge Base Search"]
            tool_checkbox_group = gr.CheckboxGroup(tool_options, label="Tools Select Options")

        with gr.Column():
            # 创建一个包含Local Knowledge Collection Select Options的Radio组件
            collection_options = ["None", "Read Existing Collection", "Create New Collection"]
            collection_checkbox_group = gr.Radio(collection_options, label="Local Knowledge Collection Select Options",
                                                 value=collection_options[0])

            collection_options = ["Openai Embedding", "HuggingFace Embedding"]
            Embedding_Model_select = gr.Radio(collection_options, label="Embedding Model Select Options",
                                              value=collection_options[0])

        with gr.Column():
            input_chunk_size = gr.Textbox(value="512", label="Input Chunk Size")
            local_data_embedding_token_max = gr.Slider(1024, 12288, step=2,
                                                       label="Embeddings Data Max Tokens", value=3072)

            # 创建一个包含Select existed Collection的Dropdown组件
            collection_name_select = gr.Dropdown(["..."], label="Select existed Collection",
                                                 value="...")
            # 为Local Knowledge Collection Select Options的Radio组件添加一个change事件，当它的值改变时，
            # 调用update_collection_name函数，并将Select existed Collection的Dropdown组件作为输出
            collection_checkbox_group.change(fn=update_collection_name, inputs=collection_checkbox_group,
                                             outputs=collection_name_select)
        with gr.Column():
            new_collection_name = gr.Textbox("", label="Input New Collection Name")
            uploaded_files = gr.File(file_count="multiple", label="Upload Files")

    temperature_num = gr.Slider(0, 1, render=False, label="Temperature")
    print_speed_step = gr.Slider(10, 20, render=False, label="Print Speed Step")

    custom_title = """
    <h1 style='text-align: center; margin-bottom: 1rem; font-family: "Courier New", monospace; 
               background: linear-gradient(135deg, #9400D3, #4B0082, #0000FF, #008000, #FFFF00, #FF7F00, #FF0000);
               -webkit-background-clip: text;
               color: transparent;'>
        RainbowGPT-Agent
    </h1>
    """

    custom_description = """
    <div style='font-size: 12px; font-family: Arial, sans-serif; text-align: right; 
                background: linear-gradient(135deg, #ff4e50, #fc913a, #fed766, #4f98ca, #4f98ca, #fc913a, #ff4e50);
                -webkit-background-clip: text;
                color: transparent;'>
        <p>How to reach me: <a href='mailto:zhujiadongvip@163.com'>zhujiadongvip@163.com</a></p>
    </div>
    """

    gr.ChatInterface(
        echo, additional_inputs=[llm_options_checkbox_group, collection_name_select, collection_checkbox_group,
                                 new_collection_name,
                                 temperature_num, print_speed_step, tool_checkbox_group, uploaded_files,
                                 Embedding_Model_select, input_chunk_size, local_data_embedding_token_max,
                                 Google_proxy],
        title=custom_title,
        description=custom_description,
        # css=".gradio-container {background-color: #f0f0f0;}",  # Add your desired background color here
    )

RainbowGPT.queue().launch(share=True)
