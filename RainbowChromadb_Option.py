import gradio as gr
import chromadb


class ChromaDBGradioUI:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=".chromadb/")
        self.collections = self.client.list_collections()
        self.create_interface()

    def create_interface(self):
        with gr.Blocks() as self.interface:
            # 下拉列表
            self.collections_combo = gr.Dropdown(choices=[collection.name for collection in self.collections],
                                                 label="选择要操作的 collection name")
            # 功能按钮
            delete_button = gr.Button("删除集合")
            # 集合信息显示
            self.collection_info_text = gr.Textbox(label="Collection 信息", interactive=False)
            # 新增刷新按钮
            refresh_button = gr.Button("刷新并且显示所有集合")
            refresh_button.click(fn=self.refresh_collections, inputs=None,
                                 outputs=[self.collection_info_text, self.collections_combo])

            # 日志信息
            self.log_text = gr.Textbox(label="操作日志", interactive=False)

            # 功能绑定
            delete_button.click(fn=self.delete_collection, inputs=self.collections_combo,
                                outputs=[self.collection_info_text, self.log_text, self.collections_combo])

        # 初始显示集合信息
        self.update_collection_info()

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
        # 删除指定的集合
        self.client.delete_collection(str(collection_name))
        updated_info, log_message = self.update_collections()
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
