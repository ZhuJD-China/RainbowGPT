import os
import logging


def get_proxy(proxy: str = None):
    """
    获取系统设置的代理。

    函数将尝试从提供的参数或环境变量中读取代理设置。
    支持的环境变量包括 all_proxy, ALL_PROXY, http_proxy, HTTP_PROXY, https_proxy, HTTPS_PROXY。
    对于不符合标准格式的代理地址，函数将进行适当的格式转换或抛出异常。

    :param proxy: 可选，显式指定的代理字符串。
    :return: 格式化后的代理字符串或None（如果未找到合适的代理设置）。
    """
    try:
        proxy = (
                proxy
                or os.environ.get("all_proxy")
                or os.environ.get("ALL_PROXY")
                or os.environ.get("http_proxy")
                or os.environ.get("HTTP_PROXY")
                or os.environ.get("https_proxy")
                or os.environ.get("HTTPS_PROXY")
        )

        if proxy:
            if proxy.startswith("socks5h://"):
                proxy = "socks5://" + proxy[len("socks5h://"):]
            elif not (proxy.startswith("http://") or proxy.startswith("https://") or proxy.startswith("socks5://")):
                raise ValueError(f"Unsupported proxy format: {proxy}")
            return proxy
        else:
            logging.warning("No proxy configuration found in environment variables.")
            return None
    except Exception as e:
        logging.error(f"Error getting proxy: {e}")
        raise


if __name__ == "__main__":
    # 测试函数
    try:
        print(get_proxy())
    except ValueError as e:
        print(f"Error: {e}")
