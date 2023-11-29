import chardet


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


# 获取用户输入的文件路径
input_csv_file = input("请输入输入文件的路径：")
detected_encoding = detect_encoding(input_csv_file)

print(f"源文件的编码格式: {detected_encoding}")

