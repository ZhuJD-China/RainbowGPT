import tkinter as tk
from tkinter import messagebox, ttk
import chromadb

class ChromaDBUI:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=".chromadb/")
        self.collections = self.client.list_collections()

        self.root = tk.Tk()
        self.root.title("ChromaDB 操作界面")
        self.set_window_size(width=400, height=300)

        self.collection_info_text = tk.Text(self.root, height=10, width=50)
        self.collection_info_text.pack()

        self.collection_name_var = tk.StringVar()
        self.collection_name_var.set(self.collections[0].name if self.collections else "")  # 设置默认值

        self.create_widgets()

    def set_window_size(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_position = (screen_width - width) // 2
        y_position = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x_position}+{y_position}")

    def create_widgets(self):
        # 下拉列表
        collections_label = tk.Label(self.root, text="选择要操作的 collection name：")
        collections_label.pack()

        collection_names = [collection.name for collection in self.collections]
        collections_combo = ttk.Combobox(self.root, textvariable=self.collection_name_var, values=collection_names, width=30)
        collections_combo.pack()

        # 功能选择
        options_label = tk.Label(self.root, text="请选择功能：")
        options_label.pack()

        delete_button = tk.Button(self.root, text="删除集合", command=self.delete_collection)
        delete_button.pack()

        # 初始化显示集合信息
        self.show_collections()

    def delete_collection(self):
        collection_name = self.collection_name_var.get()
        self.client.delete_collection(str(collection_name))
        self.update_collections()

    def update_collections(self):
        self.collections = self.client.list_collections()
        collection_names = [collection.name for collection in self.collections]
        self.collection_name_var.set(collection_names[0] if collection_names else "")  # 设置默认值
        self.update_combobox_values(collection_names)
        self.show_collections()

    def update_combobox_values(self, collection_names):
        collections_combo = self.root.children['!combobox']
        collections_combo['values'] = collection_names

    def show_all_collections(self):
        all_collections_info = "\n".join([str(collection) for collection in self.collections])
        self.collection_info_text.delete(1.0, tk.END)  # 清空Text组件内容
        self.collection_info_text.insert(tk.END, all_collections_info)

    def show_collections(self):
        collection_names = [collection.name for collection in self.collections]
        self.collection_info_text.delete(1.0, tk.END)  # 清空Text组件内容
        self.collection_info_text.insert(tk.END, "\n".join(collection_names))

if __name__ == "__main__":
    app = ChromaDBUI()
    app.root.mainloop()
