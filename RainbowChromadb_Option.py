import datetime
import io
import os
import sys

import gradio as gr
import chromadb
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
import pdfplumber
import docx


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
                # 左侧控制面板
                with gr.Column(scale=2):
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
                    # 初始化下拉列表
                    collection_names = [collection.name for collection in self.collections]
                    self.collections_combo = gr.Dropdown(
                        choices=collection_names,
                        label="Select collection name",
                        value=collection_names[0] if collection_names else None
                    )
                    delete_button = gr.Button("Delete Collection")

                # 右侧显示和使用说明
                with gr.Column(scale=3):
                    # 上半部分显示区域
                    gr.Markdown("### Refresh and Display Settings")
                    self.collection_info_text = gr.Textbox(label="Collection INFO", interactive=False, lines=5)
                    refresh_button = gr.Button("Refresh and Display All Collections")
                    self.log_text = gr.Textbox(label="Options Logs ..", interactive=False, lines=10)
                    
                    # 下半部分使用说明
                    gr.Markdown("""
                        ### 🌈 知识库管理工具使用指南
                        
                        #### 1️⃣ 创建新知识库
                        1. **基本设置**
                           - Collection Name: 输入英文名称
                           - Embedding Model: 选择向量化模型
                             * Openai Embedding: 更准确，需要API
                             * HuggingFace Embedding: 本地运行，免费
                           - Chunk Size: 文档切分大小(建议300-500)
                           - Chunk Overlap: 重叠长度(建议10-50)
                        
                        2. **文件上传**
                           - 支持格式：PDF、TXT、DOCX、MD
                           - 可多选文件
                           - 建议文件使用英文名
                        
                        #### 2️⃣ 删除知识库
                        - 从下拉菜单选择要删除的知识库
                        - 点击Delete确认删除
                        - 删除操作不可恢复，请谨慎
                        
                        #### ⚠️ 注意事项
                        - 知识库名称仅支持英文
                        - 大文件处理需要较长时间
                        - 建议定期备份重要数据
                    """)

            # 功能绑定
            refresh_button.click(fn=self.refresh_collections, inputs=None,
                                 outputs=[self.collection_info_text, self.collections_combo])
            delete_button.click(fn=self.delete_collection, inputs=self.collections_combo,
                                outputs=[self.collection_info_text, self.log_text, self.collections_combo])
            Create_button.click(fn=self.create_new_collection,
                                inputs=[self.new_collection_name, self.Embedding_Model_select,
                                        self.input_chunk_size, self.uploaded_files, self.intput_chunk_overlap],
                                outputs=[self.log_text])

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
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        base_folder = "\\data\\" + str(new_collection_name)
        save_folder = current_script_folder + f"{base_folder}_{current_time}"

        try:
            os.makedirs(save_folder, exist_ok=True)
        except Exception as e:
            response = str(e)
            for i in range(0, len(response), int(3)):
                yield response[: i + int(3)]
            return

        # 保存上传的文件
        try:
            for file in uploaded_files:
                if hasattr(file, 'name'):
                    source_file_path = file.name
                elif hasattr(file, 'orig_name'):
                    source_file_path = file.orig_name
                else:
                    source_file_path = str(file)

                file_name = os.path.basename(source_file_path)
                save_path = os.path.join(save_folder, file_name)

                # 保存文件
                if hasattr(file, 'read'):
                    content = file.read()
                    mode = 'wb' if isinstance(content, bytes) else 'w'
                    with open(save_path, mode) as target_file:
                        target_file.write(content)
                else:
                    with open(source_file_path, 'rb') as source_file:
                        with open(save_path, 'wb') as target_file:
                            target_file.write(source_file.read())

        except Exception as e:
            response = f"保存文件时发生异常：{str(e)}"
            for i in range(0, len(response), 3):
                yield response[: i + 3]
            return

        # 使用DirectoryLoader加载文档
        documents = []
        try:
            response = "开始加载PDF文件..."
            print(response)
            for i in range(0, len(response), 3):
                yield response[: i + 3]
                
            # 加载PDF文件
            pdf_loader = DirectoryLoader(
                save_folder,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                use_multithreading=True,
                show_progress=True,
                silent_errors=True
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            
            response = f"PDF文件加载完成，成功加载 {len(pdf_docs)} 个文档。"
            print(response)
            for i in range(0, len(response), 3):
                yield response[: i + 3]

            response = "开始加载TXT文件..."
            print(response)
            for i in range(0, len(response), 3):
                yield response[: i + 3]
                
            # 加载TXT文件
            text_loader = DirectoryLoader(
                save_folder,
                glob="**/*.txt",
                loader_cls=TextLoader,
                use_multithreading=True,
                show_progress=True,
                loader_kwargs={"autodetect_encoding": True},
                silent_errors=True
            )
            txt_docs = text_loader.load()
            documents.extend(txt_docs)
            
            response = f"TXT文件加载完成，成功加载 {len(txt_docs)} 个文档。"
            print(response)
            for i in range(0, len(response), 3):
                yield response[: i + 3]

            response = "开始加载DOCX文件..."
            print(response)
            for i in range(0, len(response), 3):
                yield response[: i + 3]
                
            # 加载DOCX文件
            docx_loader = DirectoryLoader(
                save_folder,
                glob="**/*.{docx,doc}",
                loader_cls=UnstructuredFileLoader,
                use_multithreading=True,
                show_progress=True,
                silent_errors=True
            )
            docx_docs = docx_loader.load()
            documents.extend(docx_docs)
            
            response = f"DOCX文件加载完成，成功加载 {len(docx_docs)} 个文档。"
            print(response)
            for i in range(0, len(response), 3):
                yield response[: i + 3]

        except Exception as e:
            response = f"加载文档时发生异常：{str(e)}"
            print(response)
            for i in range(0, len(response), 3):
                yield response[: i + 3]
            return

        if not documents:
            response = "没有成功解析任何文档内容！"
            print(response)
            for i in range(0, len(response), 3):
                yield response[: i + 3]
            return

        response = f"所有文档处理完成，共解析 {len(documents)} 个文档。开始切分文档..."
        print(response)
        for i in range(0, len(response), 3):
            yield response[: i + 3]

        # 文本切分
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=int(input_chunk_size),
            chunk_overlap=int(intput_chunk_overlap)
        )

        texts = text_splitter.split_documents(documents)
        
        response = f"文档切分完成，共生成 {len(texts)} 个文本块。开始创建向量库..."
        print(response)
        for i in range(0, len(response), 3):
            yield response[: i + 3]

        try:
            self.docsearch_db = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                collection_name=f"{new_collection_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                persist_directory=self.persist_directory
            )
            response = "知识库创建完成！"
            for i in range(0, len(response), 3):
                yield response[: i + 3]
        except Exception as e:
            response = f"创建向量库时发生错误：{str(e)}"
            for i in range(0, len(response), 3):
                yield response[: i + 3]

    def show_all_collections(self):
        # 显示所有集合前先更新集合列表
        self.refresh_collections()
        return "\n".join([collection.name for collection in self.collections])

    def refresh_collections(self):
        # 刷新集合列表
        self.collections = self.client.list_collections()
        collection_names = [collection.name for collection in self.collections]
        updated_info = "\n".join(collection_names)
        # 返回新的 Dropdown 组件
        return (
            updated_info, 
            gr.Dropdown(
                choices=collection_names,
                value=collection_names[0] if collection_names else None
            )
        )

    def delete_collection(self, collection_name):
        try:
            if not collection_name:
                raise ValueError("Please select a valid collection name.")

            self.client.delete_collection(str(collection_name))
            
            # 更新集合列表
            self.collections = self.client.list_collections()
            collection_names = [collection.name for collection in self.collections]
            updated_info = "\n".join(collection_names)
            log_message = f"Collection {collection_name} deleted successfully."
            
            # 返回新的 Dropdown 组件
            return (
                updated_info, 
                log_message, 
                gr.Dropdown(
                    choices=collection_names,
                    value=collection_names[0] if collection_names else None
                )
            )
        except Exception as e:
            log_message = f"Error: {str(e)}"
            collection_names = [collection.name for collection in self.collections]
            return (
                self.show_all_collections(),
                log_message,
                gr.Dropdown(
                    choices=collection_names,
                    value=collection_names[0] if collection_names else None
                )
            )

    def update_collections(self):
        # 更新集合列表
        self.collections = self.client.list_collections()
        updated_info = "\n".join([collection.name for collection in self.collections])
        log_message = "集合已更新"
        return updated_info, log_message

    def update_collection_info(self):
        # 更新显示的集合信息
        all_collections_info = "\n".join([str(collection) for collection in self.collections])
        return all_collections_info

    def show_all_collections(self):
        # 显示所有集合
        all_collections_info = "\n".join([collection.name for collection in self.collections])
        return all_collections_info

    def launch(self):
        return self.interface
