import pandas as pd
from googleapiclient.discovery import build
import json
import urllib.parse
import urllib.request
import requests
from bs4 import BeautifulSoup

from langchain.utilities.google_search import GoogleSearchAPIWrapper
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from dotenv import load_dotenv
import os
import requests

load_dotenv()

## Load Google API key and custom search engine ID from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')


# def set_global_proxy(proxy_url):
#     os.environ['http_proxy'] = proxy_url
#     os.environ['https_proxy'] = proxy_url


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
    - dict: Results of the Google Custom Search API.
    """
    print("google_custom_search......")
    # print("http_proxy:", os.environ['http_proxy'])
    service = build("customsearch", "v1", developerKey=api_key)
    results = service.cse().list(q=query, cx=custom_search_engine_id).execute()
    # print(results)

    # 提取标题、链接和摘要信息
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
    # 创建DataFrame
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
    Get the main content of a website.

    Parameters:
    - url (str): The URL of the website.

    Returns:
    - str: The main content of the website, or None if the request fails.
    """
    print("get_website_content.....")
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        # Extract text content from HTML
        text_content = soup.get_text(separator=' ')
        cleaned_context = text_content.replace('\n', ' ').strip()

        return cleaned_context
    else:
        print(f"Failed to retrieve content. Status code: {response.status_code}")
        return "None"


"""
google_search_results = google_custom_search("2023年12月7日新闻", GOOGLE_API_KEY, GOOGLE_CSE_ID)
print(google_search_results.to_string(index_names=False))

Google_Search = GoogleSearchAPIWrapper()
data = Google_Search.run("2023年12月7日新闻")
print(data)

kg_search_results = knowledge_graph_search('Taylor Swift', GOOGLE_API_KEY)
print("Knowledge Graph Search Results:", kg_search_results)

selenium_results = selenium_google_answer_box("以太坊的价格",
                                              "Stock_Agent/chromedriver-120.0.6099.56.0.exe")
print("Selenium Google Search Results:", selenium_results)

for link in google_search_results['Link']:
    website_content = get_website_content(link)
    if website_content:
        print("Website Content:")
        print(website_content)
"""
# set_global_proxy("http://localhost:7890")
# google_search_results = google_custom_search(
#     "2023年12月12日左右时间的有关空冷系统、空冷配件、余热发电、光热发电系统、光热发电产品类型的新闻动态", GOOGLE_API_KEY,
#     GOOGLE_CSE_ID)
# print(google_search_results)
