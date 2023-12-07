import threading
import pandas as pd
import openai
import datetime
import os
import dashscope
from dotenv import load_dotenv
import gradio as gr
import akshare as ak
import get_news_stock
import get_concept_data

load_dotenv()
# 加载互联网共享接口
# 免费授权API接口网站: https://api.chatanywhere.org/v1/oauth/free/github/render
# 转发Host1: https://api.chatanywhere.com.cn (国内中转，延时更低，推荐)
# 转发Host2: https://api.chatanywhere.cn (国外使用,国内需要全局代理)
# openai.api_base = "https://api.chatanywhere.com.cn/v1"
# 加载环境变量中的 OpenAI API 密钥
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
dashscope.api_key = DASHSCOPE_API_KEY
print(dashscope.api_key)
print(openai.api_key)

print("行业板块名称更新.............")
concept_name = get_concept_data.stock_board_concept_name_ths()


def openai_0_28_1_api_call(model="gpt-3.5-turbo-1106",
                           instruction="",
                           message="你好啊？", timestamp_str="", result=None, index=None, stock_name=None):
    try:
        print("openai_0_28_1_api_call..................")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": message}
            ]
        )
        gpt_response = response['choices'][0]['message']['content']
        gpt_file_name = f"{stock_name}_gpt_response_{timestamp_str}.txt"
        with open(gpt_file_name, 'w', encoding='utf-8') as gpt_file:
            gpt_file.write(gpt_response)
        print(f"OpenAI API 响应已保存到文件: {gpt_file_name}")
        # return gpt_response
        result[index] = gpt_response  # 将结果存储在结果列表中，使用给定的索引
    except Exception as e:
        print(f"发生异常: {response}")
        # 在这里可以添加适当的异常处理代码，例如记录异常日志或采取其他适当的措施
        result[index] = f"发生异常: {response}"


def qwen_api_call(model="qwen-72b-chat",
                  instruction="",
                  message="你好啊？", timestamp_str="", result=None, index=None, stock_name=None):
    try:
        print("qwen_api_call............................")
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": message}
        ]
        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            result_format='message',  # set the result is message format.
        )
        qwen_response = response["output"]["choices"][0]["message"]["content"]
        qwen_file_name = f"{stock_name}_qwen_response_{timestamp_str}.txt"
        with open(qwen_file_name, 'w', encoding='utf-8') as qwen_file:
            qwen_file.write(qwen_response)
        print(f"qwen API 响应已保存到文件: {qwen_file_name}")
        # return qwen_response
        result[index] = qwen_response  # 将结果存储在结果列表中，使用给定的索引
    except Exception as e:
        print(f"发生异常: {response}")
        # 在这里可以添加适当的异常处理代码，例如记录异常日志或采取其他适当的措施
        result[index] = f"发生异常: {response}"


def calculate_technical_indicators(stock_zh_a_hist_df,
                                   ma_window=5, macd_windows=(12, 26, 9),
                                   rsi_window=14, cci_window=20):
    # 丢弃NaN值
    stock_zh_a_hist_df = stock_zh_a_hist_df.dropna()

    # 检查是否有足够的数据来计算均线
    if len(stock_zh_a_hist_df) < ma_window:
        print("历史数据不足，无法计算均线。请提供更多的历史数据。")
        return None

    # 计算最小的均线
    column_name = f'MA_{ma_window}'
    stock_zh_a_hist_df[column_name] = stock_zh_a_hist_df['收盘'].rolling(window=ma_window).mean()

    # 计算MACD
    short_window, long_window, signal_window = macd_windows
    stock_zh_a_hist_df['ShortEMA'] = stock_zh_a_hist_df['收盘'].ewm(span=short_window, adjust=False).mean()
    stock_zh_a_hist_df['LongEMA'] = stock_zh_a_hist_df['收盘'].ewm(span=long_window, adjust=False).mean()
    stock_zh_a_hist_df['MACD'] = stock_zh_a_hist_df['ShortEMA'] - stock_zh_a_hist_df['LongEMA']
    stock_zh_a_hist_df['SIGNAL'] = stock_zh_a_hist_df['MACD'].ewm(span=signal_window, adjust=False).mean()

    # 计算RSI
    delta = stock_zh_a_hist_df['收盘'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_window, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    stock_zh_a_hist_df['RSI'] = 100 - (100 / (1 + rs))

    # 计算CCI
    TP = (stock_zh_a_hist_df['最高'] + stock_zh_a_hist_df['最低'] + stock_zh_a_hist_df['收盘']) / 3
    SMA = TP.rolling(window=cci_window, min_periods=1).mean()
    MAD = (TP - SMA).abs().rolling(window=cci_window, min_periods=1).mean()
    stock_zh_a_hist_df['CCI'] = (TP - SMA) / (0.015 * MAD)

    return stock_zh_a_hist_df[['日期', f'MA_{ma_window}', 'MACD', 'SIGNAL', 'RSI', 'CCI']]


def process_prompt(stock_zyjs_ths_df, stock_individual_info_em_df, stock_zh_a_hist_df, stock_news_em_df,
                   stock_individual_fund_flow_df, technical_indicators_df,
                   stock_financial_analysis_indicator_df, single_industry_df, concept_info_df):
    prompt_template = """当前股票主营业务介绍:
    {stock_zyjs_ths_df}
    
    当前股票所在的行业资金流数据:
    {single_industry_df}
    
    当前股票所在的概念板块的数据:
    {concept_info_df}
    
    当前股票基本数据:
    {stock_individual_info_em_df}
    
    当前股票历史行情数据:
    {stock_zh_a_hist_df}
    
    当前股票的K线技术指标:
    {technical_indicators_df}
    
    当前股票最近的新闻:
    {stock_news_em_df}
    
    当前股票历史的资金流动:
    {stock_individual_fund_flow_df}
    
    当前股票的财务指标数据:
    {stock_financial_analysis_indicator_df}
    
    """
    prompt_filled = prompt_template.format(stock_zyjs_ths_df=stock_zyjs_ths_df,
                                           stock_individual_info_em_df=stock_individual_info_em_df,
                                           stock_zh_a_hist_df=stock_zh_a_hist_df,
                                           stock_news_em_df=stock_news_em_df,
                                           stock_individual_fund_flow_df=stock_individual_fund_flow_df,
                                           technical_indicators_df=technical_indicators_df,
                                           stock_financial_analysis_indicator_df=stock_financial_analysis_indicator_df,
                                           single_industry_df=single_industry_df,
                                           concept_info_df=concept_info_df
                                           )
    return prompt_filled


def get_stock_data(llm_options_checkbox_group, llm_options_checkbox_group_qwen, market, symbol, stock_name,
                   start_date,
                   end_date, concept):
    instruction = "你作为A股分析专家,请详细分析市场趋势、行业前景，揭示潜在投资机会,请确保提供充分的数据支持和专业见解。"

    # 主营业务介绍 TODO 根据主营业务网络搜索相关事件报道
    stock_zyjs_ths_df = ak.stock_zyjs_ths(symbol=symbol).to_string(index=False)

    # 个股信息查询
    stock_individual_info_em_df = ak.stock_individual_info_em(symbol=symbol)
    # 提取上市时间
    list_date = stock_individual_info_em_df[stock_individual_info_em_df['item'] == '上市时间']['value'].values[0]
    # 提取行业
    industry = stock_individual_info_em_df[stock_individual_info_em_df['item'] == '行业']['value'].values[0]
    stock_individual_info_em_df = stock_individual_info_em_df.to_string(index=False)

    # 获取当前个股所在行业板块情况
    stock_sector_fund_flow_rank_df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
    single_industry_df = stock_sector_fund_flow_rank_df[stock_sector_fund_flow_rank_df['名称'] == industry]
    single_industry_df = single_industry_df.to_string(index=False)

    # 获取概念板块的数据情况
    concept_info_df = get_concept_data.stock_board_concept_info_ths(symbol=concept, stock_board_ths_map_df=concept_name)
    concept_info_df = concept_info_df.to_string(index=False)

    # 个股历史数据查询
    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date,
                                            adjust="")
    # 个股技术指标计算
    technical_indicators_df = calculate_technical_indicators(stock_zh_a_hist_df)
    stock_zh_a_hist_df = stock_zh_a_hist_df.to_string(index=False)
    technical_indicators_df = technical_indicators_df.to_string(index=False)

    # 个股新闻
    stock_news_em_df = get_news_stock.stock_news_em(symbol=symbol, pageSize=10)
    # 删除指定列
    stock_news_em_df = stock_news_em_df.drop(["文章来源", "新闻链接"], axis=1)
    stock_news_em_df = stock_news_em_df.to_string(index=False)

    # 历史的个股资金流
    stock_individual_fund_flow_df = ak.stock_individual_fund_flow(stock=symbol, market=market)
    # 转换日期列为 datetime 类型，以便进行排序
    stock_individual_fund_flow_df['日期'] = pd.to_datetime(stock_individual_fund_flow_df['日期'])
    # 按日期降序排序
    sorted_data = stock_individual_fund_flow_df.sort_values(by='日期', ascending=False)
    num_records = min(20, len(sorted_data))
    # 提取最近的至少20条记录，如果不足20条则提取所有记录
    recent_data = sorted_data.head(num_records)
    stock_individual_fund_flow_df = recent_data.to_string(index=False)

    # 财务指标
    stock_financial_analysis_indicator_df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2023")
    stock_financial_analysis_indicator_df = stock_financial_analysis_indicator_df.to_string(index=False)

    # 即时的个股资金流
    # stock_fund_flow_individual_df = ak.stock_fund_flow_individual(symbol="即时")
    # specific_stock_data = stock_fund_flow_individual_df[
    #     stock_fund_flow_individual_df['股票简称'] == stock_name].to_string(
    #     index=False)
    # 个股公司股本变动
    # stock_share_change_cninfo_df = ak.stock_share_change_cninfo(symbol=symbol, start_date=str(list_date),
    #                                                             end_date=end_date).to_string(index=False)
    # 分红配送详情
    # stock_fhps_detail_ths_df = ak.stock_fhps_detail_ths(symbol=symbol).to_string(index=False)
    # 获取高管持股数据
    # stock_ggcg_em_df = ak.stock_ggcg_em(symbol="全部")
    # stock_ggcg_em_df = stock_ggcg_em_df[stock_ggcg_em_df['名称'] == stock_name].to_string(index=False)

    # 构建最终prompt
    finally_prompt = process_prompt(stock_zyjs_ths_df, stock_individual_info_em_df, stock_zh_a_hist_df,
                                    stock_news_em_df,
                                    stock_individual_fund_flow_df, technical_indicators_df
                                    , stock_financial_analysis_indicator_df, single_industry_df, concept_info_df)
    # return finally_prompt
    user_message = (
        f"{finally_prompt}\n"
        f"请基于以上收集到的实时的真实数据，发挥你的A股分析专业知识，对未来3天该股票的价格走势做出深度预测。\n"
        f"在预测中请全面考虑主营业务、基本数据、所在行业数据、所在概念板块数据、历史行情、最近新闻以及资金流动等多方面因素。\n"
        f"给出具体的涨跌百分比数据分析总结。\n\n"
        f"以下是具体问题，请详尽回答：\n\n"
        f"1. 对最近这个股票的资金流动情况以及所在行业的资金流情况和所在概念板块的资金情况分别进行深入分析，"
        f"请详解这三个维度的资金流入或者流出的主要原因，并评估是否属于短期现象和未来的影响。\n\n"
        f"2. 基于最近财务指标数据，深刻评估公司未来业绩是否有望积极改善，可以关注盈利能力、负债情况等财务指标。"
        f"同时分析未来财务状况。\n\n"
        f"3. 是否存在与行业或公司相关的积极或者消极的消息，可能对股票价格产生什么影响？分析新闻对市场情绪的具体影响，"
        f"并评估消息的可靠性和长期影响。\n\n"
        f"4. 基于技术分析指标，如均线、MACD、RSI、CCI等，请提供更为具体的未来走势预测。"
        f"关注指标的交叉和趋势，并解读当下可能的买卖信号。\n\n"
        f"5. 在综合以上分析的基础上，向投资者推荐在未来3天内采取何种具体操作？"
        f"从不同的投资者角度明确给出买入、卖出、持有或补仓或减仓的建议，并说明理由，附上相应的止盈/止损策略。"
        f"记住给出的策略需要精确给我写出止盈位的价格，充分利用利润点，或者精确写出止损位的价格，规避亏损风险。\n\n"
        f"你可以一步一步的去思考，期待你深刻的分析，将有力指导我的投资决策。"
    )

    print(user_message)

    #
    # 获取当前时间戳字符串
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{stock_name}_{timestamp_str}.txt"  # 修改这一行，确保文件名合法
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(user_message)
    print(f"{stock_name}_已保存到文件: {file_name}")

    # 创建一个列表来存储结果
    result = [None, None]

    # 创建两个线程，分别调用不同的API，并把结果保存在列表中
    gpt_thread = threading.Thread(
        target=openai_0_28_1_api_call,
        args=(
            llm_options_checkbox_group, instruction,
            user_message, timestamp_str, result, 0, stock_name)  # 注意这里多传了两个参数，分别是列表和索引
    )
    qwen_thread = threading.Thread(
        target=qwen_api_call,
        args=(
            llm_options_checkbox_group_qwen,
            instruction,
            user_message, timestamp_str, result, 1, stock_name)  # 同上
    )

    # 把两个线程对象保存在一个列表中
    threads = [gpt_thread, qwen_thread]
    # threads = [qwen_thread]

    # 启动所有的线程
    for thread in threads:
        thread.start()

    # 等待所有的线程结束
    for thread in threads:
        thread.join()

    # 获取结果
    gpt_response = result[0]  # 通过列表和索引来访问结果
    qwen_response = result[1]

    return gpt_response, qwen_response


# Define Gradio interface for the main application
with gr.Row():
    with gr.Column():
        llm_options = ["gpt-3.5-turbo-1106", "gpt-4-1106-preview",
                       "gpt-4", "gpt-3.5-turbo-16k"]
        llm_options_checkbox_group = gr.Dropdown(llm_options, label="GPT Model Select Options",
                                                 value=llm_options[0])
        llm_options_qwen = ["qwen-72b-chat"]
        llm_options_checkbox_group_qwen = gr.Dropdown(llm_options_qwen, label="Qwen Model Select Options",
                                                      value=llm_options_qwen[0])
        # instruction = gr.Textbox(lines=1, placeholder="请输入模型instruction指令",
        #                          value="需要你发挥强大的A股分析专家能力。请详细分析市场趋势、行业前景，揭示潜在投资机会")
        market = gr.Textbox(lines=1, placeholder="请输入股票市场（sz或sh，示例：sz）", value="sz")
        symbol = gr.Textbox(lines=1, placeholder="请输入股票代码(示例股票代码:002665)", value="002665")
        stock_name = gr.Textbox(lines=1, placeholder="请输入股票名称(示例股票名称:首航高科): ", value="首航高科")
        start_date = gr.Textbox(lines=1, placeholder="请输入K线历史数据查询起始日期（YYYYMMDD，示例：20230805）: ",
                                value="20230805")
        end_date = gr.Textbox(lines=1, placeholder="请输入K线历史数据结束日期（YYYYMMDD，示例：20231206）: ",
                              value="20231206")
        concept = gr.Textbox(lines=1, placeholder="请输入当前股票所属概念板块名称(示例：光热发电): ", value="光热发电")
    with gr.Column():
        gpt_response = gr.Textbox(label="GPT Response")
        qwen_response = gr.Textbox(label="QWEN Response")

custom_title = "StockGPT Analysis"
custom_description = """
<p>How to reach me: <a href='mailto:zhujiadongvip@163.com'>zhujiadongvip@163.com</a></p>
"""
# Define Gradio interface for the main application
iface = gr.Interface(
    fn=get_stock_data,
    inputs=[llm_options_checkbox_group, llm_options_checkbox_group_qwen, market, symbol, stock_name,
            start_date,
            end_date, concept],
    outputs=[gpt_response, qwen_response],
    title=custom_title,
    description=custom_description
)

# Launch Gradio interface for the main application
iface.queue().launch(share=True)
