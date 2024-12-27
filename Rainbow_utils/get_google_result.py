import httplib2
import pandas as pd
from googleapiclient.discovery import build
import json
import urllib.parse
import urllib.request
import requests
from bs4 import BeautifulSoup
import winreg
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from dotenv import load_dotenv
import os
import requests

load_dotenv()

## Load Google API key and custom search engine ID from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')


def get_windows_proxy():
    proxy_settings = {"http": None, "https": None}
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
            proxy_enable = winreg.QueryValueEx(key, "ProxyEnable")[0]
            proxy_server = winreg.QueryValueEx(key, "ProxyServer")[0]

            if proxy_enable:
                # 处理可能的多个代理设置
                if ';' in proxy_server:
                    for proxy in proxy_server.split(';'):
                        if proxy.startswith("http="):
                            proxy_settings["http"] = proxy.split('=')[1]
                        elif proxy.startswith("https="):
                            proxy_settings["https"] = proxy.split('=')[1]
                else:
                    proxy_settings["http"] = proxy_settings["https"] = proxy_server

    except Exception as e:
        print(f"Error retrieving proxy settings: {e}")
    proxies = {
        "http": "127.0.0.1:10809",
        "https": "127.0.0.1:10809"
    }
    return proxies
    # return proxy_settings


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
    http_proxy = proxies.get('http')
    https_proxy = proxies.get('https')

    # Create an Http object with proxy support
    proxy_info = httplib2.ProxyInfo(
        httplib2.socks.PROXY_TYPE_HTTP,
        http_proxy.split(':')[0],
        int(http_proxy.split(':')[1]),
        proxy_rdns=True
    ) if http_proxy else None

    http = httplib2.Http(proxy_info=proxy_info) if proxy_info else httplib2.Http()

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
    Get the main content of a website using system proxy settings.

    Parameters:
    - url (str): The URL of the website.

    Returns:
    - str: The main content of the website, or None if the request fails.
    """
    print("get_website_content.....")
    print("url:", url)

    # Get system proxy settings
    proxies = get_windows_proxy()

    try:
        response = requests.get(url, proxies=proxies)
        if response.status_code == 200:
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            # Extract text content from HTML
            text_content = soup.get_text(separator=' ')
            cleaned_context = text_content.replace('\n', ' ').strip()

            print("get_website_content.....done")
            return cleaned_context
        else:
            print(f"Failed to retrieve content. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


print(get_windows_proxy())
if __name__ == "__main__":
    # print(get_windows_proxy())

    google_search_results, google_search_results2 = google_custom_search("2023年12月7日新闻", GOOGLE_API_KEY,
                                                                         GOOGLE_CSE_ID)
    # print(google_search_results.to_string(index_names=False))
    print(google_search_results, google_search_results2)

    for link in google_search_results:
        website_content = get_website_content(link)
        if website_content:
            print("Website Content:")
            print(website_content)

    # Google_Search = GoogleSearchAPIWrapper()
    # data = Google_Search.run("2023年12月7日新闻")
    # print(data)
    #
    # kg_search_results = knowledge_graph_search('Taylor Swift', GOOGLE_API_KEY)
    # print("Knowledge Graph Search Results:", kg_search_results)
    #
    # selenium_results = selenium_google_answer_box("以太坊的价格",
    #                                               "Stock_Agent/chromedriver-120.0.6099.56.0.exe")
    # print("Selenium Google Search Results:", selenium_results)
    #
