import os


def convert_files(source_directory, target_directory, source_extension, target_extension):
    for root, _, files in os.walk(source_directory):
        for filename in files:
            if filename.endswith(source_extension) or source_extension == "*":
                source_path = os.path.join(root, filename)
                target_path = os.path.join(target_directory, os.path.splitext(filename)[0] + target_extension)

                try:
                    with open(source_path, "r", encoding="utf-8") as source_file:
                        source_content = source_file.read()
                except UnicodeDecodeError:
                    # Skip files that cannot be read as UTF-8 text
                    print(f"Skipped non-text file: {filename}")
                    continue

                with open(target_path, "w", encoding="utf-8") as target_file:
                    target_file.write(source_content)
                os.remove(source_path)
                print(f"Converted {filename} to {os.path.basename(target_path)}")


source_directory = input("请输入源目录路径：")
target_directory = input("请输入目标目录路径：")
source_extension = input("请输入改前格式（包括点号，例如 .mdx），或输入 * 选择所有文件：")
target_extension = input("请输入改后格式（包括点号，例如 .md）：")

convert_files(source_directory, target_directory, source_extension, target_extension)
