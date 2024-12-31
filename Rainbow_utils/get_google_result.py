import httplib2
import pandas as pd
from googleapiclient.discovery import build
import json
import urllib.parse
import urllib.request
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from dotenv import load_dotenv
import os
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

## Load Google API key and custom search engine ID from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')


def get_windows_proxy():
    try:
        # 首先尝试检查代理是否可用
        test_proxies = {
            "http": "127.0.0.1:10809",
            "https": "127.0.0.1:10809"
        }
        
        print("Testing proxy connection...")
        print(f"Current proxy settings: {test_proxies}")
        
        # 测试代理连接
        test_response = requests.get(
            "https://www.google.com", 
            proxies=test_proxies, 
            timeout=5, 
            verify=False
        )
        print(f"Proxy test successful! Status code: {test_response.status_code}")
        return test_proxies
    except requests.exceptions.ProxyError as e:
        print(f"Proxy error: {e}")
        print("Proxy server is not responding. Please check if your proxy service is running.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        print("Failed to connect using proxy. Switching to direct connection.")
        return None
    except Exception as e:
        print(f"Unexpected error while testing proxy: {e}")
        return None


def get_published_date(item):
    # 获取结果项的发布时间，这里以'pagemap'中的'metatags'为例
    if 'pagemap' in item and 'metatags' in item['pagemap']:
        metatags = item['pagemap']['metatags']
        if metatags and 'og:article:published_time' in metatags:
            return metatags['og:article:published_time']

    # 如果没有找到时间信息，返回一个较早的日期
    return '1970-01-01T00:00:00Z'


def google_custom_search(query, api_key=GOOGLE_API_KEY, custom_search_engine_id=GOOGLE_CSE_ID):
    """
    Perform a Google Custom Search.

    Parameters:
    - query (str): The search query.
    - api_key (str): Google API key.
    - custom_search_engine_id (str): Google custom search engine ID.

    Returns:
    - tuple: Tuple containing two lists, the first with the links and the second with the merged titles and snippets.
    """
    print("google_custom_search......")
    print("query:", query) 

    # Automatically detect and set system proxy
    proxies = get_windows_proxy()
    
    # Create an Http object with proxy support if proxies are available
    http = httplib2.Http()
    if proxies:
        http_proxy = proxies.get('http')
        if http_proxy:
            proxy_info = httplib2.ProxyInfo(
                httplib2.socks.PROXY_TYPE_HTTP,
                http_proxy.split(':')[0],
                int(http_proxy.split(':')[1]),
                proxy_rdns=True
            )
            http = httplib2.Http(proxy_info=proxy_info)

    # Setup Google Custom Search API service
    service = build("customsearch", "v1", developerKey=api_key, http=http)
    results = service.cse().list(q=query, cx=custom_search_engine_id).execute()

    # Extract titles, links, and snippets
    link_data = []
    data_without_link = []
    search_results = results.get('items', [])
    for result in search_results:
        title = result.get('title', '')
        link = result.get('link', '')
        snippet = result.get('snippet', '')

        link_data.append(link)
        merged_content = title + ' ' + snippet
        data_without_link.append(merged_content)

    print("google_custom_search.....done")
    return link_data, data_without_link


def knowledge_graph_search(query, api_key):
    """
    Perform a Knowledge Graph Search.

    Parameters:
    - query (str): The search query.
    - api_key (str): Google API key.

    Returns:
    - list: Results of the Knowledge Graph Search API.
    """
    print("knowledge_graph_search......")
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'query': query,
        'limit': 10,
        'indent': True,
        'key': api_key,
    }
    url = service_url + '?' + urllib.parse.urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read())
    results = [(element['result']['name'], element['resultScore']) for element in response['itemListElement']]
    return results


def extract_google_answer(driver, query):
    """
    Extracts information from Google's answer box using a given query.

    Parameters:
    - driver (webdriver): The Selenium WebDriver instance.
    - query (str): The search query.

    Returns:
    - str: The extracted information from the Google answer box.
    """
    print("extract_google_answer......")
    try:
        formatted_query = query.replace(' ', '+')
        driver.get(f'http://www.google.com/search?q={formatted_query}')
        answers = []
        max_answers = 3

        for y in range(200, 300, 40):
            script = "return document.elementFromPoint(arguments[0], arguments[1]);"
            element = driver.execute_script(script, 350, y)

            if element and element.text:
                answers.append(element.text)
                if len(answers) >= max_answers:
                    break

        result = '\n'.join(answers)
        result = result[:150] if len(result) > 150 else result
        return result
    except Exception as e:
        print(f"Error occurred: {e}")
        return "None"


def selenium_google_answer_box(query, chrome_driver_path):
    """
    Use Selenium to extract information from the Google answer box.

    Parameters:
    - query (str): The search query.
    - chrome_driver_path (str): Path to the ChromeDriver executable.

    Returns:
    - list: Extracted information from the Google answer box.
    """
    # print("selenium_google_answer_box......", os.environ['http_proxy'])
    print("selenium_google_answer_box......")
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-extensions')
    options.add_argument('--headless')
    options.add_argument('--disable-images')
    options.add_argument('--disable-plugins')
    options.add_argument('--disable-gpu')
    options.page_load_strategy = 'eager'

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    results = extract_google_answer(driver, query)
    driver.quit()

    if results == None or results == "":
        return "None"
    else:
        return results


def get_website_content(url):
    """
    使用多种爬虫技术获取网页内容
    """
    print("\nStarting website content retrieval...")
    print(f"Target URL: {url}")

    # Get system proxy settings
    proxies = get_windows_proxy()
    print(f"Using proxy settings: {proxies}")

    # 增强的请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'DNT': '1'
    }

    def extract_text_with_beautifulsoup(html_content):
        """使用BeautifulSoup提取文本"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 移除不需要的标签
        for tag in soup(['script', 'style', 'meta', 'noscript', 'header', 'footer', 'nav']):
            tag.decompose()
        
        # 优先提取主要内容区域
        main_content = None
        priority_tags = ['article', 'main', 'div[role="main"]', '.content', '#content', '.post', '.article']
        
        for selector in priority_tags:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if main_content:
            # 提取主要内容区域的文本
            text = main_content.get_text(separator=' ', strip=True)
        else:
            # 如果没找到主要内容区域，提取所有段落文本
            paragraphs = []
            for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = p.get_text(strip=True)
                if len(text) > 20:  # 过滤掉太短的段落
                    paragraphs.append(text)
            text = ' '.join(paragraphs)
        
        return text

    def extract_text_with_readability(html_content):
        """使用readability-lxml提取文本"""
        try:
            from readability import Document
            doc = Document(html_content)
            return doc.summary()
        except Exception as e:
            print(f"Readability extraction failed: {e}")
            return None

    def clean_text(text):
        """清理和规范化文本"""
        import re
        if not text:
            return ""
        
        # 替换多个空白字符为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\w\s.,!?;:()\'\"，。！？；：（）]', '', text)
        # 移除重复的标点符号
        text = re.sub(r'([.,!?;:。！？；：])\1+', r'\1', text)
        return text.strip()

    try:
        # 尝试使用代理获取内容
        print("Attempting to fetch content with proxy...")
        response = requests.get(
            url,
            proxies=proxies if proxies else None,
            headers=headers,
            verify=False,
            timeout=10,
            allow_redirects=True
        )
        
        if response.status_code == 200:
            # 确保正确的编码
            response.encoding = response.apparent_encoding
            html_content = response.text
            
            # 尝试多种提取方法
            content = None
            
            # 1. 使用 BeautifulSoup 提取
            content = extract_text_with_beautifulsoup(html_content)
            
            # 2. 如果BeautifulSoup提取的内容太少，尝试readability
            if not content or len(content) < 100:
                readability_content = extract_text_with_readability(html_content)
                if readability_content and len(readability_content) > len(content or ""):
                    content = readability_content
            
            # 清理和规范化文本
            if content:
                cleaned_content = clean_text(content)
                if len(cleaned_content) > 50:  # 确保提取的内容有意义
                    print("Content successfully retrieved and processed")
                    return cleaned_content
            
            print("Failed to extract meaningful content")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        # 尝试不使用代理
        try:
            print("Attempting to fetch without proxy...")
            response = requests.get(
                url,
                headers=headers,
                verify=False,
                timeout=10,
                allow_redirects=True
            )
            if response.status_code == 200:
                response.encoding = response.apparent_encoding
                html_content = response.text
                content = extract_text_with_beautifulsoup(html_content)
                if not content or len(content) < 100:
                    content = extract_text_with_readability(html_content)
                
                if content:
                    return clean_text(content)
        except Exception as e:
            print(f"Direct connection also failed: {e}")
            return None
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

    return None


print(get_windows_proxy())
if __name__ == "__main__":
    # 测试代理连接
    print("=== Testing Proxy Connection ===")
    proxy_result = get_windows_proxy()
    print(f"Final proxy settings: {proxy_result}")
    print("==============================\n")

    google_search_results, google_search_results2 = google_custom_search("2023年12月7日新闻", GOOGLE_API_KEY,
                                                                         GOOGLE_CSE_ID)
    print("Search URLs:", google_search_results)
    
    # 遍历所有URL直到获取到有效内容
    content = None
    for link in google_search_results:
        print(f"\nTrying URL: {link}")
        content = get_website_content(link)
        if content and len(content.strip()) > 0:  # 检查内容是否为空
            print("Successfully retrieved content from:", link)
            print("Content length:", len(content))
            print("Content preview:", content[:200] + "...")
            break  # 找到有效内容后退出循环
        else:
            print(f"No valid content from {link}, trying next URL...")
    
    if not content:  # 如果所有URL都失败了
        print("\nAll primary URLs failed. Trying backup URLs...")
        # 尝试第二组搜索结果
        for link in google_search_results2:
            print(f"\nTrying backup URL: {link}")
            content = get_website_content(link)
            if content and len(content.strip()) > 0:
                print("Successfully retrieved content from backup URL:", link)
                print("Content length:", len(content))
                print("Content preview:", content[:200] + "...")
                break
            else:
                print(f"No valid content from backup URL {link}")
    
    if not content:
        print("\nFailed to retrieve content from all URLs")
    else:
        print("\nFinal content successfully retrieved")
