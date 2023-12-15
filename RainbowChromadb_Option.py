import datetime
import io
import os
import sys

import gradio as gr
import chromadb
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma


class ChromaDBGradioUI:
    def __init__(self):
        self.path = ".chromadb/"
        self.persist_directory = self.path
        self.client = chromadb.PersistentClient(path=self.path)
        self.collections = self.client.list_collections()
        self.docsearch_db = None
        self.create_interface()

    def create_interface(self):
        with gr.Blocks() as self.interface:
            # 使用行和列来组织布局
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Create Collection Settings")
                    with gr.Row():
                        self.new_collection_name = gr.Textbox("", label="New Collection Name")
                        self.Embedding_Model_select = gr.Radio(["Openai Embedding", "HuggingFace Embedding"],
                                                               label="Embedding Model Select",
                                                               value="HuggingFace Embedding")
                    with gr.Row():
                        self.input_chunk_size = gr.Textbox(value="512", label="Create Chunk Size")
                        self.intput_chunk_overlap = gr.Textbox(value="16", label="Create Chunk overlap Size")
                    self.uploaded_files = gr.File(file_count="multiple", label="Upload Files")
                    Create_button = gr.Button("Create Collection")

                    gr.Markdown("### Delete Collection Settings")
                    # 下拉列表
                    self.collections_combo = gr.Dropdown(choices=[collection.name for collection in self.collections],
                                                         label="Select collection name",
                                                         value="...")
                    # 功能按钮
                    delete_button = gr.Button("Delete Collection")

                with gr.Column():
                    gr.Markdown("### Refresh and Display Settings")
                    # 集合信息显示
                    self.collection_info_text = gr.Textbox(label="Collection INFO", interactive=False)
                    # 新增刷新按钮
                    refresh_button = gr.Button("Refresh and Display All Collections")
                    refresh_button.click(fn=self.refresh_collections, inputs=None,
                                         outputs=[self.collection_info_text, self.collections_combo])
                    # 日志信息
                    self.log_text = gr.Textbox(label="Options Logs ..", interactive=False)

            # 功能绑定
            delete_button.click(fn=self.delete_collection, inputs=self.collections_combo,
                                outputs=[self.collection_info_text, self.log_text, self.collections_combo])
            Create_button.click(fn=self.create_new_collection,
                                inputs=[self.new_collection_name, self.Embedding_Model_select,
                                        self.input_chunk_size, self.uploaded_files, self.intput_chunk_overlap],
                                outputs=[self.log_text]
                                )

        # 初始显示集合信息
        self.update_collection_info()

    def create_new_collection(self, new_collection_name, Embedding_Model_select, input_chunk_size, uploaded_files,
                              intput_chunk_overlap):
        response = f"{Embedding_Model_select} 模型加载中....."
        print(response)
        for i in range(0, len(response), int(3)):
            yield response[: i + int(3)]

        if Embedding_Model_select in ["Openai Embedding", "", None]:
            self.embeddings = OpenAIEmbeddings()
            self.embeddings.show_progress_bar = True
            self.embeddings.request_timeout = 20
            self.Embedding_Model_select_global = 0
        elif Embedding_Model_select == "HuggingFace Embedding":
            self.embeddings = HuggingFaceEmbeddings(cache_folder="models")
            self.Embedding_Model_select_global = 1

        if new_collection_name == None or new_collection_name == "":
            response = "新知识库的名字没有写，创建中止！"
            print(response)
            for i in range(0, len(response), int(3)):
                yield response[: i + int(3)]
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
            response = str(e)
            for i in range(0, len(response), int(3)):
                yield response[: i + int(3)]
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
            response = str(e)
            for i in range(0, len(response), int(3)):
                yield response[: i + int(3)]
            print(f"保存文件时发生异常：{e}")

        # 设置向量存储相关配置
        response = "开始转换文件夹中的所有数据成知识库........"
        print(response)

        loader = DirectoryLoader(str(save_folder), show_progress=True,
                                 use_multithreading=True,
                                 silent_errors=True)

        documents = loader.load()
        if documents == None:
            response = "文件读取失败！" + str(save_folder)
            for i in range(0, len(response), int(3)):
                yield response[: i + int(3)]
            print(response)
            return

        response = str(documents)
        if len(response) == 0:
            response = "文件读取失败！" + str(save_folder)
            for i in range(0, len(response), int(3)):
                yield response[: i + int(3)]
            return
        else:
            response = "文档数据长度为： " + str(documents.__len__()) + response
            for i in range(0, len(response), len(response) // 5):
                yield response[: i + (len(response) // 5)]
            print(response)

        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=int(input_chunk_size),
                                              chunk_overlap=int(intput_chunk_overlap))

        texts = text_splitter.split_documents(documents)
        print(texts)
        response = str(texts)
        for i in range(0, len(response), len(response) // 5):
            yield response[: i + (len(response) // 5)]
        print("after split documents len= ", texts.__len__())
        response = "切分之后文档数据长度为：" + str(texts.__len__()) + " 数据开始写入词向量库....."
        for i in range(0, len(response), int(3)):
            yield response[: i + int(3)]
        print(response)

        # Collection does not exist, create it
        self.docsearch_db = Chroma.from_documents(documents=texts, embedding=self.embeddings,
                                                  collection_name=str(new_collection_name + "_" + current_time),
                                                  persist_directory=self.persist_directory,
                                                  Embedding_Model_select=self.Embedding_Model_select_global)

        response = "知识库建立完毕！！"
        for i in range(0, len(response), int(3)):
            yield response[: i + int(3)]
        print(response)

    def show_all_collections(self):
        # 显示所有集合前先更新集合列表
        self.refresh_collections()
        return "\n".join([collection.name for collection in self.collections])

    def refresh_collections(self):
        # 刷新集合列表
        self.collections = self.client.list_collections()
        updated_info = "\n".join([collection.name for collection in self.collections])
        return updated_info, gr.Dropdown.update(choices=[collection.name for collection in self.collections])

    def delete_collection(self, collection_name):
        try:
            # Ensure a valid collection name is selected
            if collection_name == "...":
                raise ValueError("Please select a valid collection name.")

            # Delete the specified collection
            self.client.delete_collection(str(collection_name))

            # Update the collections and log message
            updated_info, log_message = self.update_collections()
            log_message = f"Collection {collection_name} deleted successfully."

        except ValueError as e:
            # Handle specific known errors
            log_message = f"Error: {str(e)}"
            updated_info = self.show_all_collections()

        except Exception as e:
            # Handle any other exceptions
            log_message = f"Unexpected error: {str(e)}"
            updated_info = self.show_all_collections()

        return updated_info, log_message, gr.Dropdown.update(
            choices=[collection.name for collection in self.collections])

    def update_collections(self):
        # 更新集合列表
        self.collections = self.client.list_collections()
        updated_info = "\n".join([collection.name for collection in self.collections])
        log_message = "集合已更新"
        return updated_info, log_message

    def update_collection_info(self):
        # 更新显示的集合信息
        all_collections_info = "\n".join([str(collection) for collection in self.collections])
        self.collection_info_text.update(value=all_collections_info)

    def show_all_collections(self):
        # 显示所有集合
        all_collections_info = "\n".join([collection.name for collection in self.collections])
        return all_collections_info

    def launch(self):
        return self.interface
