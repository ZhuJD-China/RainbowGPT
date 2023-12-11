import re
import langid
import tiktoken


def detect_language(text):
    """
    Detects the language of the given text using the langid library.

    Args:
    - text (str): The text to detect the language.

    Returns:
    - Tuple containing language code and confidence.
    """
    language, confidence = langid.classify(text)
    return language, confidence


def filter_chinese_english_punctuation(text):
    """
    Filters Chinese, English, numbers, punctuation, and spaces from the text.

    Args:
    - text (str): The text to filter.

    Returns:
    - Filtered text string.
    """
    # Replace newline characters with two spaces
    text = text.replace('\n', '  ')

    # 使用正则表达式保留中文、英文、数字、标点符号和空格
    # [^\u4e00-\u9fa5a-zA-Z0-9.,;:!? ] matches any character that is not Chinese, not English, not a number, not punctuation, and not a space
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,;:!? ]', '', text)
    return text


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """
    Returns the number of tokens in a text string.

    Args:
    - string (str): The text string to calculate the number of tokens.
    - encoding_name (str): The encoding name used for tokenization.

    Returns:
    - Number of tokens in the text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_string_to_max_tokens(input_string, max_tokens, tokenizer_name, step_size=5):
    """
    Truncates the input string to a maximum number of tokens with caching and a step size.

    Args:
    - input_string (str): The input string to truncate.
    - max_tokens (int): The maximum number of tokens allowed.
    - tokenizer_name (str): The name of the tokenizer.
    - step_size (int): The step size for increasing truncation length.

    Returns:
    - Truncated string.
    """
    # Start from the step size, gradually increase truncation length, ensuring not to exceed the maximum number of tokens
    for truncation_length in range(1, len(input_string) + 1, step_size):
        truncated_string = input_string[:truncation_length]

        # Calculate the number of tokens for the current truncated string (using a caching function)
        current_tokens = num_tokens_from_string(truncated_string, tokenizer_name)

        # If the current token count exceeds the maximum token count, return the previous truncated string
        if current_tokens > max_tokens:
            return input_string[:truncation_length - step_size]

    # If the entire string meets the requirements, return the original string
    return input_string
