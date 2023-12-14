#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Desc: 个股新闻数据
https://so.eastmoney.com/news/s?keyword=%E4%B8%AD%E5%9B%BD%E4%BA%BA%E5%AF%BF&pageindex=1&searchrange=8192&sortfiled=4
"""
import json
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import pandas as pd
import re
from urllib.parse import quote


def stock_news_em(symbol: str = "601628", pageSize: int = 10, chrome_driver_path="") -> pd.DataFrame:
    """
    东方财富-个股新闻-最近 100 条新闻
    https://so.eastmoney.com/news/s?keyword=%E4%B8%AD%E5%9B%BD%E4%BA%BA%E5%AF%BF&pageindex=1&searchrange=8192&sortfiled=4
    :param symbol: 股票代码
    :type symbol: str
    :return: 个股新闻
    :rtype: pandas.DataFrame
    """
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])  # 禁止打印日志
    options.add_argument('--ignore-certificate-errors')
    # linux下所需参数
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-extensions')
    options.add_argument('headless')
    # 当前文件夹里chromedriver路径

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)

    # 构建请求参数
    params = {
        "uid": "",
        "keyword": symbol,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": pageSize,
                "preTag": "<em>",
                "postTag": "</em>"
            }
        }
    }

    # 转换为 JSON 字符串
    params_json = json.dumps(params)

    # 进行 URL 编码
    encoded_params = quote(params_json, safe='')
    # 构建完整的请求URL
    url = (f'https://search-api-web.eastmoney.com/search/jsonp?'
           f'cb=jQuery35108613950799967576_1701396301284&param={encoded_params}&_=1701396301285')

    driver.get(url)
    data_text = driver.page_source
    # print(data_text)
    pattern = re.compile(r'"bizCode"(.*?)\)</pre>', re.DOTALL)
    data_re_list = pattern.findall(data_text)
    # print(data_re_list)
    data_json = json.loads(
        '{"bizCode"' + data_re_list[0]
    )
    temp_df = pd.DataFrame(data_json["result"]["cmsArticleWebOld"])
    temp_df.rename(
        columns={
            "date": "发布时间",
            "mediaName": "文章来源",
            "code": "-",
            "title": "新闻标题",
            "content": "新闻内容",
            "url": "新闻链接",
            "image": "-",
        },
        inplace=True,
    )
    temp_df["关键词"] = symbol
    temp_df = temp_df[
        [
            "关键词",
            "新闻标题",
            "新闻内容",
            "发布时间",
            "文章来源",
            "新闻链接",
        ]
    ]
    temp_df["新闻标题"] = (
        temp_df["新闻标题"]
        .str.replace(r"\(<em>", "", regex=True)
        .str.replace(r"</em>\)", "", regex=True)
    )
    temp_df["新闻标题"] = (
        temp_df["新闻标题"]
        .str.replace(r"<em>", "", regex=True)
        .str.replace(r"</em>", "", regex=True)
    )
    temp_df["新闻内容"] = (
        temp_df["新闻内容"]
        .str.replace(r"\(<em>", "", regex=True)
        .str.replace(r"</em>\)", "", regex=True)
    )
    temp_df["新闻内容"] = (
        temp_df["新闻内容"]
        .str.replace(r"<em>", "", regex=True)
        .str.replace(r"</em>", "", regex=True)
    )
    temp_df["新闻内容"] = temp_df["新闻内容"].str.replace(r"\u3000", "", regex=True)
    temp_df["新闻内容"] = temp_df["新闻内容"].str.replace(r"\r\n", " ", regex=True)
    return temp_df


def save_to_excel(df: pd.DataFrame, symbol: str):
    """
    将 DataFrame 写入 Excel 文件，并以股票代码加时间戳字符串为后缀保存
    :param df: DataFrame
    :param symbol: 股票代码
    :type df: pd.DataFrame
    :type symbol: str
    """
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    filename_with_timestamp = f"{symbol}_{timestamp_str}.xlsx"
    df.to_excel(filename_with_timestamp, index=False)
    print(f"数据已保存至 {filename_with_timestamp}")


if __name__ == "__main__":
    symbol = "首航高科"
    stock_news_em_df = stock_news_em(symbol=symbol)
    save_to_excel(stock_news_em_df, symbol)
