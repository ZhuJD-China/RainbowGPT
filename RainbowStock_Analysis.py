import threading
import pandas as pd
import openai
import datetime
import os
import dashscope
from dotenv import load_dotenv
import gradio as gr
import akshare as ak
from Rainbow_utils import get_news_stock
from Rainbow_utils import get_concept_data
from Rainbow_utils import get_google_result
from datetime import datetime
import time
import re
from Rainbow_utils.get_tokens_cal_filter import filter_chinese_english_punctuation, \
    truncate_string_to_max_tokens
import concurrent.futures
import requests
import PyPDF2
from io import BytesIO
from openai import OpenAI
from Rainbow_utils.model_config_manager import ModelConfigManager
from langchain_community.chat_models import ChatBaichuan
from langchain_core.messages import HumanMessage
from Rainbow_utils.baichuan_api import BaichuanAPI


class RainbowStock_Analysis:
    def __init__(self):
        """初始化类"""
        self.load_dotenv()
        self.initialize_variables()
        self.create_interface()

    def load_dotenv(self):
        """加载环境变量"""
        load_dotenv()

    def initialize_variables(self):
        """初始化变量"""
        try:
            # 初始化模型配置管理器
            self.model_manager = ModelConfigManager()
            
            # 加载概念名称数据
            try:
                self.concept_name = pd.read_csv('./Rainbow_utils/concept_name.csv')
            except Exception as e:
                print(f"Warning: Failed to load concept_name.csv: {str(e)}")
                self.concept_name = None
                
        except Exception as e:
            print(f"Error in initialize_variables: {str(e)}")
            raise

    def openai_async_api_call(self, instruction="You are a helpful assistant.",
                             message="", timestamp_str="", result=None, index=None, stock_name=None):
        """
        使用全局配置的模型进行 API 调用
        """
        try:
            print("Starting API call...")
            
            # 获取当前活动的模型配置
            config = self.model_manager.get_active_config()
            if not config:
                raise ValueError("No active model configuration found")

            # Handle Baichuan model
            if config.model_name == "Baichuan-192K":
                try:
                    # 创建Baichuan API客户端实例
                    baichuan_client = BaichuanAPI(api_key=config.api_key)
                    
                    # 合并instruction和message
                    combined_message = f"{instruction}\n\n{message}" if instruction else message
                    
                    # 构建消息列表
                    messages = [
                        {"role": "user", "content": combined_message}
                    ]
                    
                    # 调用Baichuan API
                    gpt_response = baichuan_client.chat_completion(
                        messages=messages,
                        temperature=config.temperature,
                        stream=True  # 使用流式输出
                    )
                    
                    print(f"Baichuan API Response: {gpt_response}")
                    
                except Exception as baichuan_error:
                    error_detail = f"Baichuan API Error: {str(baichuan_error)}"
                    raise Exception(error_detail)
                    
            else:
                # OpenAI API调用保持不变
                client = OpenAI(
                    api_key=config.api_key,
                    base_url=config.api_base
                )
                
                response = client.chat.completions.create(
                    model=config.model_name,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": message}
                    ],
                    temperature=config.temperature
                )
                gpt_response = response.choices[0].message.content
            
            # Save response to file
            gpt_file_name = f"{stock_name}_gpt_response_{timestamp_str}.txt"
            gpt_file_name = "./logs/" + gpt_file_name
            with open(gpt_file_name, 'w', encoding='utf-8') as gpt_file:
                gpt_file.write(gpt_response)
            print(f"API response saved to file: {gpt_file_name}")
            
            if result is not None and index is not None:
                result[index] = gpt_response
                
            return gpt_response
            
        except Exception as e:
            error_message = f"API call failed: {str(e)}"
            print(error_message)
            
            # Save detailed error log
            error_file_name = f"{stock_name}_error_{timestamp_str}.txt"
            error_file_name = "./logs/" + error_file_name
            try:
                with open(error_file_name, 'w', encoding='utf-8') as error_file:
                    error_file.write(error_message)
                    error_file.write("\n\nDebug Information:\n")
                    error_file.write(f"Model Type: {config.model_name}\n")
                    error_file.write(f"API Key Length: {len(config.api_key)}\n")
                    error_file.write(f"Message Length: {len(message)}\n")
                    if hasattr(e, '__dict__'):
                        error_file.write(f"Error attributes: {str(e.__dict__)}\n")
            except Exception as file_error:
                print(f"Failed to write error log: {str(file_error)}")
            
            if result is not None and index is not None:
                result[index] = error_message
                
            return error_message

    def calculate_technical_indicators(self, stock_zh_a_hist_df,
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

    def process_prompt(self, stock_zyjs_ths_df, stock_individual_info_em_df, stock_zh_a_hist_df, stock_news_em_df,
                       stock_individual_fund_flow_df, technical_indicators_df,
                       stock_financial_analysis_indicator_df, single_industry_df, concept_info_df):
        prompt_template = """当前股票主营业务和产业的相关的历史动态:
        {stock_zyjs_ths_df}

        当前股票所在的行业资金数据:
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

    def format_date(self, input_date):
        # 将输入日期字符串解析为 datetime 对象
        date_object = datetime.strptime(input_date, "%Y%m%d")

        # 将 datetime 对象格式化为指定的日期字符串
        formatted_date = date_object.strftime("%Y年%m月%d日")

        return formatted_date

    # 函数来提取日期并转换为datetime对象
    def extract_and_convert_date(self, text):
        # 使用正则表达式来匹配日期格式 "年-月-日" 或 "月 日, 年"
        match = re.search(r'(\d{4})[ 年](\d{1,2})[ 月](\d{1,2})[ 日]|(\w{3}) (\d{1,2}), (\d{4})', text)
        if match:
            if match.group(1):  # 匹配 "年-月-日" 格式
                return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            else:  # 匹配 "月 日, 年" 格式
                month = datetime.strptime(match.group(4), '%b').month
                return datetime(int(match.group(6)), month, int(match.group(5)))
        return None

    def extract_text_from_pdf(self, pdf_url):
        try:
            # Send a GET request to download the PDF
            response = requests.get(pdf_url)

            # Check if the request was successful
            if response.status_code == 200:
                # Read the PDF content from the response
                with BytesIO(response.content) as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)

                    # Extract text from each page
                    pdf_text = [page.extract_text() for page in reader.pages]

                    # Combine the text from all pages
                    full_text = "\n".join(filter(None, pdf_text))
                    return full_text
            else:
                return "Failed to retrieve the PDF file."
        except Exception as e:
            return f"Error: {e}"

    def is_pdf_url(self, url):
        return url.lower().endswith('.pdf')

    def process_link(self, link):
        """Function to process each link."""
        truncated_text = None
        if self.is_pdf_url(link):
            result_text = self.extract_text_from_pdf(link)
            website_content = filter_chinese_english_punctuation(result_text)
            truncated_text = truncate_string_to_max_tokens(website_content,
                                                           300,
                                                           "cl100k_base",
                                                           step_size=150)
            return truncated_text
        else:
            website_content = get_google_result.get_website_content(link)
            if website_content:
                website_content = filter_chinese_english_punctuation(website_content)
                truncated_text = truncate_string_to_max_tokens(website_content,
                                                               300,
                                                               "cl100k_base",
                                                               step_size=150)
            return truncated_text
        return None

    def get_stock_data(self, market, symbol, stock_name,
                       start_date, end_date, concept, http_proxy):
        """获取股票数据并进行分析"""
        instruction = "你作为A股分析家,请详细分析市场趋势、行业前景，揭示潜在投资机会,请确保提供充分的数据支持和专业见解。"

        # 主营业务介绍-根据主营业务网络搜索相关事件报道
        # get_google_result.set_global_proxy(http_proxy)

        stock_zyjs_ths_df = ak.stock_zyjs_ths(symbol=symbol)
        formatted_date = self.format_date(end_date)
        IN_Q = str(formatted_date) + "的有关" + stock_zyjs_ths_df['产品类型'].to_string(index=False) + "产品类型的新闻动态"
        IN_Q = stock_name
        print("IN_Q:",IN_Q)
        custom_search_link, data_title_Summary = get_google_result.google_custom_search(IN_Q)

        # 提取每个文本片段的日期并存储在列表中，同时保留对应的链接
        dated_snippets_with_links = []
        for snippet, link in zip(data_title_Summary, custom_search_link):
            date = self.extract_and_convert_date(snippet)
            if date:
                dated_snippets_with_links.append((date, snippet, link))
        # 按日期对列表进行排序
        dated_snippets_with_links.sort(key=lambda x: x[0], reverse=True)
        # 提取前三个文本片段及其对应的链接
        first_three_snippets_with_links = dated_snippets_with_links[:2]
        # 将这三个文本片段转换为字符串，并提取对应的链接
        first_three_snippets = " ".join([snippet for _, snippet, _ in first_three_snippets_with_links])
        sorted_links = [link for _, _, link in first_three_snippets_with_links]
        # Using ThreadPoolExecutor
        link_detail_res = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map the function to the links and execute in parallel
            future_to_link = {executor.submit(self.process_link, link): link for link in sorted_links}
            for future in concurrent.futures.as_completed(future_to_link):
                result = future.result()
                if result:
                    link_detail_res.append(result)
        # Concatenate the strings in the list into a single string
        link_datial_string = '\n'.join(link_detail_res)

        stock_zyjs_ths_df = first_three_snippets + " " + link_datial_string

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
        concept_info_df = get_concept_data.stock_board_concept_info_ths(symbol=concept,
                                                                        stock_board_ths_map_df=self.concept_name)
        concept_info_df = concept_info_df.to_string(index=False)

        # 个股历史数据查询
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date,
                                                adjust="")
        # 个股技术指标计算
        technical_indicators_df = self.calculate_technical_indicators(stock_zh_a_hist_df)
        stock_zh_a_hist_df = stock_zh_a_hist_df.to_string(index=False)
        technical_indicators_df = technical_indicators_df.to_string(index=False)

        # 个股新闻
        stock_news_em_df = get_news_stock.stock_news_em(symbol=symbol, pageSize=10,
                                                        chrome_driver_path="Rainbow_utils/chromedriver.exe")
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

        # 构建最终prompt
        finally_prompt = self.process_prompt(stock_zyjs_ths_df, stock_individual_info_em_df, stock_zh_a_hist_df,
                                             stock_news_em_df,
                                             stock_individual_fund_flow_df, technical_indicators_df,
                                             stock_financial_analysis_indicator_df, single_industry_df,
                                             concept_info_df)
        
        user_message = (
            f"{finally_prompt}\n"
            f"请基于以上收集到的实时的真实数据，发挥你的A股分析专业知识，对未来3天该股票的价格走势做出深度预测。\n"
            f"在预测中请全面考虑主营业务、基本数据、所在行业数据、所在概念板块数据、历史行情、最近新闻以及资金流动等多方面因素。\n"
            f"给出具体的涨跌百分比数据分析总结。\n\n"
            f"以下是具体问题，请详尽回答：\n\n"
            f"1.对当前股票主营业务和产业的相关的历史动态进行分析行业走势。"
            f"2. 对最近这个股票的资金流动情况以及所在行业的资金情况和所在概念板块的资金情况分别进行深入分析，"
            f"请详解这三维度的资金流入或者流出的主要原因，并评估是否属于短期现象和未来的影响。\n\n"
            f"3. 基于最近财务指标数据，深刻评估公司未来业绩是否有望积极改善，可以关注盈利能力、负债情况等财务指标。"
            f"同时分析未来财务状况。\n\n"
            f"4. 是否存在与行业或公司相关的积极或者消极的消息，可能对股票价格产生什么影响？分析新闻对市场情绪的具体影响，"
            f"评估消息的可靠性和长期影响。\n\n"
            f"5. 基于技术分析指标，如均线、MACD、RSI、CCI等，请提供更为具体的未来走势预测。"
            f"关注指标的交叉和趋势，并解读当下可能的买卖信号。\n\n"
            f"6. 在综合以上分析的基础上，向投资者推荐在未来3天内采取何种具体操作？"
            f"从不同的投资者角度明确给出买入、卖出、持有或补仓或减仓的建议，并说明理由，附上相应的止盈/止损策略。"
            f"记住给出的策略需要精确给我写出止盈位的价格，充分利用利润点，或者精确写出止损位的价格，规避亏损风险。\n\n"
            f"你可以一步一步的去思考，期待你对接下来几天的股票走势和价格预测进行深刻的分析，将有力指导我的投资决策。"
        )

        # 保存用户消息到文件
        timestamp_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        file_name = f"{stock_name}_{timestamp_str}.txt"
        file_name = "./logs/" + file_name
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(user_message)
        print(f"{stock_name}_已保存到文件: {file_name}")

        # 直接调用 OpenAI API
        response = self.openai_async_api_call(
            instruction=instruction,
            message=user_message,
            timestamp_str=timestamp_str,
            stock_name=stock_name
        )

        return response

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(theme=gr.themes.Soft()) as self.interface:
            # 添加标题和说明
            gr.Markdown("""
            # 🌈 RainbowGPT Stock Analysis
            
            ## 📊 功能介绍
            本工具使用AI技术对A股股票进行深度分析，提供全面的投资建议和市场洞察。
            
            ### 🔍 分析维度
            1. 主营业务和产业动态分析 2. 多维度资金流向分析 3. 财务指标深度解读 4. 市场情绪和新闻影响评估 5. 技术指标综合分析6. 具体投资建议和策略
            
            ### 📝 使用说明
            1. 填写股票基本信息（市场、代码、名称） 2. 设置数据查询时间范围 3. 输入股票所属概念板块 4. 点击提交获取分析报告
            """)
            
            with gr.Row():
                # 左侧：输入区域
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 🔧 基础设置")
                        http_proxy = gr.Textbox(
                            value="http://localhost:10809",
                            label="HTTP代理设置",
                            info="用于Google搜索，如不需要可空"
                        )
                    
                    with gr.Group():
                        gr.Markdown("### 📈 股票信息")
                        with gr.Row():
                            market = gr.Dropdown(
                                choices=["sh", "sz"],
                                label="交易市场",
                                value="sh",
                                info="上海证券交易所(sh) 或 深圳证券交易所(sz)"
                            )
                            symbol = gr.Textbox(
                                label="股票代码",
                                placeholder="例如：600839",
                                value="600839",
                                info="6位数字代码"
                            )
                        
                        stock_name = gr.Textbox(
                            label="股票名称",
                            placeholder="例如：四川长虹",
                            value="四川长虹",
                            info="请输入完整股票名称"
                        )
                    
                    with gr.Group():
                        gr.Markdown("### 📅 时间范围")
                        with gr.Row():
                            start_date = gr.Textbox(
                                label="开始日期",
                                placeholder="YYYYMMDD",
                                value="20240805",
                                info="历史数据查询起始日期"
                            )
                            end_date = gr.Textbox(
                                label="结束日期",
                                placeholder="YYYYMMDD",
                                value="20241202",
                                info="历史数据查询结束日期"
                            )
                    
                    with gr.Group():
                        gr.Markdown("### 🏷️ 概念板块")
                        concept = gr.Textbox(
                            label="概念板块",
                            placeholder="例如：机器人概���",
                            value="机器人概念",
                            info="股票所属的主要概念板块"
                        )
                    
                    # 提交按钮
                    submit_button = gr.Button(
                        "📊 开始分析",
                        variant="primary",
                        scale=1
                    )
                
                # 右侧：输出区域
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### 📑 分析报告")
                        response = gr.Textbox(
                            label="AI 分析结果",
                            show_label=False,
                            lines=30,
                            max_lines=50,
                            show_copy_button=True
                        )
                    
                    with gr.Group():
                        gr.Markdown("""
                        ### ⚠️ 免责声明
                        1. 本工具提供的分析仅供参考，不构成投资建议
                        2. 投资有风险，入市需谨慎
                        3. 使用者应对自己的投资决策负责
                        
                        ### 📮 联系方式
                        如有问题或建议，请联系：[zhujiadongvip@163.com](mailto:zhujiadongvip@163.com)
                        """)

            # 绑定提交事件
            submit_button.click(
                fn=self.get_stock_data,
                inputs=[
                    market, symbol, stock_name,
                    start_date, end_date, concept, http_proxy
                ],
                outputs=[response]
            )

    def launch(self):
        return self.interface
