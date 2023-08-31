import chromadb

client = chromadb.PersistentClient(path=".chromadb/")
print(client.list_collections())


def main():
    while True:
        collection_name = input("请输入要操作的 collection name（输入 'exit' 退出）：")

        if collection_name == 'exit':
            break

        print("请选择功能：")
        print("1. 创建集合（create_collection）")
        print("2. 删除集合（delete_collection）")
        print("3. 获取或创建集合（get_or_create_collection）")
        print("4. 返回上一级")

        choice = input("请输入选项（1-4）：")

        if choice == "1":
            client.create_collection(str(collection_name))
            print(client.list_collections())
        elif choice == "2":
            client.delete_collection(str(collection_name))
            print(client.list_collections())
        elif choice == "3":
            client.get_or_create_collection(str(collection_name))
            print(client.list_collections())
        elif choice == "4":
            continue  # 重新进入集合名称的输入循环
        else:
            print("无效的选项，请重新输入。")


if __name__ == "__main__":
    main()
