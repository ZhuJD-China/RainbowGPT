import re
import langid


def detect_language(text):
    # 使用 langid 库检测文本的语言
    language, confidence = langid.classify(text)
    return language, confidence


def filter_chinese_english_punctuation(text):
    # 将换行符替换为两个空格
    text = text.replace('\n', '  ')

    # 使用正则表达式保留中文、英文、数字、标点符号和空格
    # [^\u4e00-\u9fa5a-zA-Z0-9.,;:!? ] 表示匹配非中文、非英文、非数字、非标点符号和非空格的所有字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,;:!? ]', '', text)
    return text
