import os
import urllib.request
import urllib.parse
import json
import datetime
import tushare as ts
from datetime import timedelta
import openai
import requests
import urllib.request
import ssl

# 设置OpenAI API密钥
openai.api_key = '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'


def get_Stock_History_White_Yellow_Line(code, day_his):
    host = 'https://ali-stock.showapi.com'
    path = '/timeline'
    method = 'GET'
    appcode = '871f666510024d6ea7fe9b04dc7d6580'
    querys = 'code=' + str(code) + '&day=' + str(day_his)
    bodys = {}
    url = host + path + '?' + querys

    request = urllib.request.Request(url)
    request.add_header('Authorization', 'APPCODE ' + appcode)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    response = urllib.request.urlopen(request, context=ctx)
    content = response.read()
    if content:
        return content


def get_macd_rsi_kdj(code, start, end):
    contents = []
    paths = ['/macd', '/rsi', '/stockKDJ']
    for path in paths:
        host = 'https://ali-stock.showapi.com'
        path = path
        method = 'GET'
        appcode = '871f666510024d6ea7fe9b04dc7d6580'
        querys = 'code=' + str(code) + '&end=' + str(end) + '&fqtype=bfq&start=' + str(start) + ''
        bodys = {}
        url = host + path + '?' + querys

        request = urllib.request.Request(url)
        request.add_header('Authorization', 'APPCODE ' + appcode)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        response = urllib.request.urlopen(request, context=ctx)
        content = response.read()
        if content:
            contents.append(content)

    if contents:
        return contents


def get_k_lines(code, start, end):
    host = 'https://ali-stock.showapi.com'
    path = '/sz-sh-stock-history'
    appcode = '871f666510024d6ea7fe9b04dc7d6580'
    querys = 'begin=' + str(start) + '&code=' + str(code) + '&end=' + str(end) + '&type=hfq'
    url = host + path + '?' + querys

    request = urllib.request.Request(url)
    request.add_header('Authorization', 'APPCODE ' + appcode)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    response = urllib.request.urlopen(request, context=ctx)
    content = response.read()

    if content:
        return content


def get_news(keyword):
    host = 'https://jisunews.market.alicloudapi.com'
    path = '/news/search'
    appcode = '871f666510024d6ea7fe9b04dc7d6580'
    url = host + path
    query = {"keyword": keyword}
    headers = {'Authorization': 'APPCODE ' + appcode, 'Content-Type': 'application/json; charset=UTF-8'}

    params = urllib.parse.urlencode(query)
    full_url = url + "?" + params
    request = urllib.request.Request(full_url, headers=headers)
    response = urllib.request.urlopen(request)
    content = response.read().decode('utf-8')
    return content


def write_to_files(directory, content, max_chars=4000):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(0, len(content), max_chars):
        with open(os.path.join(directory, f'news_{i // max_chars}.txt'), 'w', encoding='utf-8') as f:
            f.write(content[i:i + max_chars])


def analyze_and_predict(directory, content):
    # 提供给ChatGPT的提示
    prompt = f"考虑以下的新闻内容和最近的股票价格走势，请给出未来5天股票价格走势预测的涨跌百分比,并作为短线投资者的给出以下建议：买入，卖出，持有，补仓：\n\n{content}\n\n分析结果："

    with open(directory + '/prompt_TOTAL.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)

    print(
        "--------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    # 发送请求给ChatGPT进行分析和预测
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "system", "content": "请模拟中国A股的分析大师."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    # 解析ChatGPT的回复
    if 'choices' in response and len(response['choices']) > 0:
        analysis_result = response['choices'][0]['message']['content']
        return analysis_result
    else:
        return "无法获取分析和预测结果。"


# 获得最近的交易数据
ts.set_token('536fdad957268f38abb4ced920d6cf501c00dbd1e060ba2d3774f1e2')
pro = ts.pro_api()  # 初始化接口


def main():
    stocks = {
        # '002544.SZ': '普天科技',
        # '002315.SZ': '焦点科技',
        # '002037.SZ': '保利联合',
        '002665.SZ': '首航高科',
        '601996.SH': '丰林集团',
    }

    date_str = datetime.datetime.now().strftime('%Y%m%d')
    day_to = input("请输入要查询过去多少天的历史价格：(格式是30，默认值是30): ") or "30"
    macd_rsi_kdj_start = input("请输入macd rsi kdj的起始时间：(默认值是20230601): ") or "20230601"
    macd_rsi_kdj_end = input("请输入macd rsi kdj的结束时间：(默认值是20230630): ") or "20230704"
    daily_k_lines_start = input("请输入日线查询的起始时间：(默认值是2023-06-01): ") or "2023-06-01"
    daily_k_lines_end = input("请输入日线查询的结束时间：(默认值是2023-06-30): ") or "2023-07-04"
    day_H_W_Y_line = input("请输入多少天的股票历史白黄线：(默认值是1): ") or "1"

    for code, name in stocks.items():
        directory = os.path.join(f"{date_str}_{code}_{name}")
        try:

            os.makedirs(directory, exist_ok=True)  # 创建文件夹，如果文件夹已存在则不会引发错误

            content = get_news(name)
            write_to_files(directory, content)

            rsi_kdj_contents = get_macd_rsi_kdj(code.split('.')[0], macd_rsi_kdj_start, macd_rsi_kdj_end)
            with open(directory + '/' + str(name) + '_rsi_kdj_contents.txt', 'w') as f:
                f.write(str(rsi_kdj_contents))

            day_data = get_k_lines(code.split('.')[0], daily_k_lines_start, daily_k_lines_end)
            with open(directory + '/' + str(name) + '_daily_k_lines_contents.txt', 'w') as f:
                f.write(str(day_data))

            Stock_History_White_Yellow_Line = get_Stock_History_White_Yellow_Line(code.split('.')[0], day_H_W_Y_line)
            with open(directory + '/' + str(name) + '_Stock_History_White_Yellow_Line.txt', 'w') as f:
                f.write(str(Stock_History_White_Yellow_Line))



        except Exception as e:
            print(f"获取新闻[{name}]出错: {str(e)}")
            continue

        start_date = (datetime.datetime.now() - timedelta(days=int(day_to))).strftime('%Y%m%d')

        try:

            data = pro.daily(ts_code=code, start_date=start_date, end_date=date_str)

            file_name = f"{code}_{name}_historical_data.txt"

            file_path = os.path.join(directory, file_name)
            data.to_csv(file_path, index=False)


        except Exception as e:
            print(f"导出股票[{name}]的历史数据出错: {str(e)}")
            continue

        news_file_path = os.path.join(directory, 'news_0.txt')
        with open(news_file_path, 'r', encoding='utf-8') as f:
            news_content = f.read()

        try:
            recent_price = data['close'].iloc[0]
            past_price = data['close'].iloc[-1]

            price_trend = '增长' if recent_price >= past_price else '下降'
            percentage_change = abs(recent_price - past_price) / past_price * 100

            price_trend_desc_exp = f"在过去的15天里，{name}的股价呈现{price_trend}趋势，变动百分比约为{percentage_change:.2f}%。"
            price_trend_desc = f"在过去的 , {str(day_to)} , 天里股票的历史数据是以下这些内容: \n{data.to_string(index=False)}\n"
            price_trend_desc = price_trend_desc + price_trend_desc_exp

        except Exception as e:
            # 处理异常情况的代码
            recent_price = "没用查到"
            print("发生了异常:", str(e))
            price_trend_desc = f" "

        combined_content = f"新闻内容：{news_content}\n\n最近股价：{recent_price}\n\n{price_trend_desc}\n\n 这支股票macd、rsi、kdj的数据是：{rsi_kdj_contents}" \
                           f"\n\n 并且股票日线数据如下：{rsi_kdj_contents}\n\n" \
                           f"并且这支股票的股票历史白黄线如下：{Stock_History_White_Yellow_Line}"

        analysis_result = analyze_and_predict(directory, combined_content)
        print("综合分析和预测结果：", {code}, "-", {name}, analysis_result)


if __name__ == "__main__":
    main()
