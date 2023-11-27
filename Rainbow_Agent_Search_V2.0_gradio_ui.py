import datetime
import time
import chromadb
import openai
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from loguru import logger
from langchain.callbacks import FileCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from transformers import GPT2Tokenizer
import gradio as gr
from typing import Iterable
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from langchain.retrievers import BM25Retriever, EnsembleRetriever


class Seafoam(Base):
    def __init__(
            self,
            *,
            primary_hue: colors.Color | str = colors.emerald,
            secondary_hue: colors.Color | str = colors.blue,
            neutral_hue: colors.Color | str = colors.gray,
            spacing_size: sizes.Size | str = sizes.spacing_md,
            radius_size: sizes.Size | str = sizes.radius_md,
            text_size: sizes.Size | str = sizes.text_lg,
            font: fonts.Font
                  | str
                  | Iterable[fonts.Font | str] = (
                    fonts.GoogleFont("Quicksand"),
                    "ui-sans-serif",
                    "sans-serif",
            ),
            font_mono: fonts.Font
                       | str
                       | Iterable[fonts.Font | str] = (
                    fonts.GoogleFont("IBM Plex Mono"),
                    "ui-monospace",
                    "monospace",
            ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )


seafoam = Seafoam()

# 加载环境变量中的 OpenAI API 密钥
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
# 打印 API 密钥
print(OPENAI_API_KEY)
logfile = "Rainbow_Agent_Search_V2.0_gradio_ui.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)

persist_directory = ".chromadb/"
client = chromadb.PersistentClient(path=persist_directory)
collections = client.list_collections()
list_collections_name = []
for collection in collections:
    collection_name = collection.name
    list_collections_name.append(collection_name)

# 创建 ChatOpenAI 实例作为底层语言模型
llm = None
llm_name_global = None
# llm = ChatOpenAI(temperature=float(temperature_num_global), model="gpt-3.5-turbo-16k-0613")
embeddings = None
# embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings()
Embedding_Model_select_global = 0
temperature_num_global = 0

# 文档切分的长度
input_chunk_size_global = None
# 本地知识库嵌入token max
local_data_embedding_token_max_global = None

# 在文件顶部定义docsearch_db
docsearch_db = None

human_input_global = None

# Local_Search Prompt模版
local_search_template = """
你作为一个强大的AI问答和知识库内容总结分析的专家。
可以通过以下双引号内的知识库内容进行分析问答:

“{combined_text}”

如果无法回答问题则回复说无法找到答案，并且回复所搜索到的知识库分析总结。
如果可以回答问题则根据知识库和问题进行一定的思考后，回复给出针对问题最精确的回答。

我的问题是: {human_input}

"""

# 全局工具列表创建
tools = []


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
    )
    # 使用预训练的gpt2分词器
    tokenizers = GPT2Tokenizer.from_pretrained("gpt2")

    if Embedding_Model_select_global == 0:
        print("OpenAIEmbeddings")
        # 结合基础检索器+Embedding上下文压缩
        chroma_retriever = docsearch_db.as_retriever(search_kwargs={"k": 30})
        # # 获取变量的内存地址并打印
        # address = id(docsearch_db)
        # print("docsearch_db变量的内存地址:", hex(address))
        # 将压缩器和文档转换器串在一起
        splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
        relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, redundant_filter, relevant_filter]
        )
        compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                               base_retriever=chroma_retriever)
        compressed_docs = compression_retriever.get_relevant_documents(question, tools=tools)

        # 设置最大尝试次数
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                # 将稀疏检索器（如 BM25）与密集检索器（如嵌入相似性）相结合
                # 初始化 bm25 检索器和 faiss 检索器
                bm25_retriever = BM25Retriever.from_documents(compressed_docs)
                bm25_retriever.k = 5
                # 初始化 ensemble 检索器
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
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
        print("HuggingFaceEmbeddings")
        chroma_retriever = docsearch_db.as_retriever(search_kwargs={"k": 30})
        retrieved_docs = chroma_retriever.get_relevant_documents(question)
        bm25_retriever = BM25Retriever.from_documents(retrieved_docs)
        bm25_retriever.k = 2
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
        tokens = tokenizers.encode(cleaned_context, add_special_tokens=False)
        if total_toknes + len(tokens) <= (int(local_data_embedding_token_max_global)):
            cleaned_matches.append(cleaned_context)
            total_toknes += len(tokens)
        else:
            last_index = index
            break
    # 将清理过的匹配项组合合成一个字符串
    combined_text = " ".join(cleaned_matches)

    answer = local_chain.predict(combined_text=combined_text, human_input=human_input_global)
    return answer


Local_Search_tool = Tool(
    name="Local_Search",
    func=ask_local_vector_db,
    description="""
        可以通过本地知识数据库库尝试寻找问答案。
        注意你需要提出非常有针对性准确的问题和回答。
        """
)

Google_Search = GoogleSearchAPIWrapper()
Google_Search_tool = Tool(
    name="Google_Search",
    func=Google_Search.run,
    description="""
        若本地知识库没有答案，或者问题中需要网络搜索的时候都可以使用这个互联网搜索引擎工具进行信息查询,尝试直接找到问题答案。 
        将搜索到的按照问题的相关性和时间进行排序，并且你必须严格参照搜索到的资料和你自己的认识结合进行回答！
        如果搜索到一样的数据不要重复再搜索！
        注意你需要提出非常有针对性准确的问题和回答。
        """
)

tools_temp = load_tools(["llm-math"], llm=ChatOpenAI(model="gpt-3.5-turbo-16k"))
tools.append(tools_temp[0])


# bing - search
# python_repl


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

    # 设置代理（替换为你的代理地址和端口）
    proxy_url = str(Google_proxy)
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url

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
    tools_temp = load_tools(["llm-math"], llm=ChatOpenAI(model="gpt-3.5-turbo-16k"))
    tools.append(tools_temp[0])

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
        response = "切分之后文档数据长度为：" + str(texts.__len__())
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

        # TODO
        # collections = client.list_collections()
        # list_collections_name.clear()
        # for collection in collections:
        #     collection_name = collection.name
        #     list_collections_name.append(collection_name)
        # collection_name_select = gr.Dropdown(list_collections_name, label="Select existed Collection")
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

                # 获取变量的内存地址并打印
                address = id(docsearch_db)
                print("Read docsearch_db变量的内存地址:", hex(address))

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
        response = "读取知识库但是没有打开知识库搜索工具！失去本地知识库搜索能力！使用模型本身记忆回答！"
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


with gr.Blocks(theme=seafoam) as RainbowGPT:
    # TODO
    # collections = client.list_collections()
    # list_collections_name.clear()
    # for collection in collections:
    #     collection_name = collection.name
    #     list_collections_name.append(collection_name)

    with gr.Row():
        with gr.Column():
            # 创建一个包含选项的多选框组
            llm_options = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview",
                           "gpt-4-vision-preview"]
            llm_options_checkbox_group = gr.Dropdown(llm_options, label="LLM Model Select Options",
                                                     value=llm_options[0])
            Google_proxy = gr.Textbox(value="http://localhost:7890", label="Google Http Proxy")
            tool_options = ["Google Search", "Local Knowledge Base Search"]
            tool_checkbox_group = gr.CheckboxGroup(tool_options, label="Tools Select Options")

        with gr.Column():
            collection_options = ["None", "Read Existing Collection", "Create New Collection"]
            collection_checkbox_group = gr.Radio(collection_options, label="Local Knowledge Collection Select Options",
                                                 value=collection_options[0])

            collection_options = ["Openai Embedding", "HuggingFace Embedding"]
            Embedding_Model_select = gr.Radio(collection_options, label="Embedding Model Select Options",
                                              value=collection_options[0])

        with gr.Column():
            input_chunk_size = gr.Textbox(value="1024", label="Input Chunk Size")
            local_data_embedding_token_max = gr.Slider(8000, 13428, step=1,
                                                       label="Local Data Max Tokens")
            collection_name_select = gr.Dropdown(list_collections_name, label="Select existed Collection",
                                                 value="...")
        with gr.Column():
            new_collection_name = gr.Textbox("", label="Input New Collection Name")
            uploaded_files = gr.File(file_count="multiple", label="Upload Files")

    temperature_num = gr.Slider(0, 1, render=False, label="Temperature")
    print_speed_step = gr.Slider(10, 20, render=False, label="Print Speed Step")

    gr.ChatInterface(
        echo, additional_inputs=[llm_options_checkbox_group, collection_name_select, collection_checkbox_group,
                                 new_collection_name,
                                 temperature_num, print_speed_step, tool_checkbox_group, uploaded_files,
                                 Embedding_Model_select, input_chunk_size, local_data_embedding_token_max,
                                 Google_proxy],
        title="RainbowGPT-Agent",
        description="How to reach me: zhujiadongvip@163.com",
        # css=".gradio-container {background-color: red}",
    )

RainbowGPT.queue().launch(share=True)
