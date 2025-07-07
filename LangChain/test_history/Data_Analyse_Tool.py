from transformers import pipeline
import torch

MODEL_PATH = "E:/Pycharm/LangChain/embedding"

try:
    # 创建嵌入 pipelinefrom langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
    # from langchain_openai import ChatOpenAI
    # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    # from langchain_community.chat_models import ChatOpenAI
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # import numpy as np
    # from datetime import datetime, timedelta
    # import textwrap
    # import re
    # import os
    # from longbridge.openapi import QuoteContext, Config, SubType, Period, AdjustType, CalcIndex
    # from langchain.tools import Tool
    # from langchain_community.vectorstores import FAISS
    # import numpy as np
    # from datetime import datetime
    # from Embedding_Vector import Creat_Embeddings
    # from dotenv import load_dotenv
    #
    # load_dotenv()
    #
    # # 初始化Longbridge配置
    # config = Config(
    #     app_key=os.getenv("LONGPORT_APP_KEY"),
    #     app_secret=os.getenv("LONGPORT_APP_SECRET"),
    #     access_token=os.getenv("LONGPORT_ACCESS_TOKEN")
    # )
    #
    # # 配置路径
    # MODEL_PATH = "E:/Pycharm/LangChain/embedding"
    # FAISS_INDEX_PATH = "E:/Pycharm/LangChain/FAISS_Stock_Data"
    # INDEX_NAME = "hk_stock_index"
    #
    #
    # # 创建自定义嵌入对象
    # embedding = Creat_Embeddings()
    #
    # # 股票名称到代码映射
    # name_to_code = {
    #     "腾讯": "00700.HK",
    #     "阿里巴巴": "09988.HK",
    #     "美团": "03690.HK",
    #     "小米": "01810.HK",
    #     "粤港湾": "01396.HK",
    #     "京东": "09618.HK",
    #     "百度": "09888.HK",
    #     "网易": "09999.HK",
    #     "比亚迪": "01211.HK",
    #     "中国平安": "02318.HK",
    # }
    #
    #
    # # 加载 FAISS 向量库
    # def load_faiss_index():
    #     """加载 FAISS 向量库"""
    #     if not os.path.exists(os.path.join(FAISS_INDEX_PATH, f"{INDEX_NAME}.faiss")):
    #         print(f"未存储{INDEX_NAME}相关数据")
    #         return None
    #
    #     try:
    #         db = FAISS.load_local(FAISS_INDEX_PATH, embedding, index_name=INDEX_NAME, allow_dangerous_deserialization=True)
    #         print(f"已加载股票数据索引，包含 {db.index.ntotal} 个文档")
    #         return db
    #     except Exception as e:
    #         print(f"加载索引失败: {str(e)}")
    #         return None
    #
    # # 初始化向量库
    # vector_db = load_faiss_index()
    #
    #
    #
    # # 1. 基础搜索工具
    # def stock_search_tool(query: str, k: int = 5) -> str:
    #     """在向量库中搜索相关股票数据"""
    #     if not vector_db:
    #         return "股票数据索引未加载，请先加载数据"
    #
    #     results = vector_db.similarity_search(query, k=k)
    #
    #     output = []
    #     for i, doc in enumerate(results):
    #         stock_name = doc.metadata.get("stock_name", "未知股票")
    #         stock_code = doc.metadata.get("stock_code", "未知代码")
    #         update_time = doc.metadata.get("update_time", "未知时间")
    #
    #         # 格式化文本摘要
    #         content = textwrap.fill(doc.page_content, width=80)
    #
    #         output.append(
    #             f"🔍 {i + 1}. {stock_name} ({stock_code})\n"
    #             f"🕒 更新时间: {update_time}\n"
    #             f"📝 摘要: {content}\n"
    #             f"{'-' * 40}"
    #         )
    #
    #     return "\n".join(output)
    #
    # # 2. 趋势分析工具
    # def trend_analysis_tool(stock_names: str, days: int = 5) -> str:
    #     """
    #     分析股票趋势，返回文本报告和图表路径
    #     输入: 股票名称（多个用逗号分隔），分析天数
    #     输出: 趋势分析报告和图表文件路径
    #     """
    #     # 1. 设置中文字体
    #     plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
    #     plt.rcParams['axes.unicode_minus'] = False
    #
    #     stock_list = [name.strip() for name in stock_names.split(",")]
    #
    #     if not vector_db:
    #         return "股票数据索引未加载，请先加载数据"
    #
    #     # 收集所有相关文档
    #     all_docs = []
    #     for stock in stock_list:
    #         if stock not in name_to_code:
    #             continue
    #
    #         # 搜索该股票的所有文档
    #         query = f"{stock} {name_to_code[stock]}"
    #         docs = vector_db.similarity_search(query, k=10)
    #         all_docs.extend(docs)
    #
    #     if not all_docs:
    #         return "未找到相关股票数据"
    #
    #     # 解析K线数据
    #     kline_data = {}
    #     for doc in all_docs:
    #         stock_code = doc.metadata.get("stock_code")
    #         if not stock_code:
    #             continue
    #
    #         # 从文档内容中提取K线数据
    #         content = doc.page_content
    #         kline_matches = re.findall(r"最近K线:([^:]+):开(\d+\.?\d*)/收(\d+\.?\d*)，", content)
    #
    #         if not kline_matches:
    #             continue
    #
    #         # 组织K线数据
    #         if stock_code not in kline_data:
    #             kline_data[stock_code] = {
    #                 "dates": [],
    #                 "opens": [],
    #                 "closes": [],
    #                 "stock_name": doc.metadata.get("stock_name", stock_code)
    #             }
    #
    #         for match in kline_matches:
    #             date_str, open_price, close_price = match
    #             kline_data[stock_code]["dates"].append(date_str)
    #             kline_data[stock_code]["opens"].append(float(open_price))
    #             kline_data[stock_code]["closes"].append(float(close_price))
    #
    #     if not kline_data:
    #         return "未找到K线数据"
    #
    #     # 创建趋势图表
    #     plt.figure(figsize=(12, 8))
    #
    #     for stock_code, data in kline_data.items():
    #         # 只取最近days天的数据
    #         dates = data["dates"][-days:]
    #         closes = data["closes"][-days:]
    #
    #         if len(dates) < 2:
    #             continue
    #
    #         # 计算移动平均
    #         moving_avg = np.convolve(closes, np.ones(3) / 3, mode='valid')
    #
    #         plt.plot(dates, closes, 'o-', label=f"{data['stock_name']} ({stock_code})")
    #         plt.plot(dates[1:-1], moving_avg, '--', alpha=0.5)
    #
    #     plt.title(f"股票价格趋势 ({days}天)")
    #     plt.xlabel("日期")
    #     plt.ylabel("收盘价")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.xticks(rotation=45)
    #     plt.tight_layout()
    #
    #     # 保存图表
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     chart_path = f"stock_trend_{timestamp}.png"
    #     plt.savefig(chart_path)
    #     plt.close()
    #
    #     # 生成分析报告
    #     report = f"📈 股票趋势分析报告 ({days}天)\n\n"
    #     report += f"分析股票: {', '.join(stock_list)}\n"
    #     report += f"趋势图表已保存至: {chart_path}\n\n"
    #
    #     # 添加简要分析
    #     report += "分析摘要:\n"
    #     for stock_code, data in kline_data.items():
    #         if len(data["closes"]) < 2:
    #             continue
    #
    #         latest_close = data["closes"][-1]
    #         prev_close = data["closes"][-2] if len(data["closes"]) > 1 else latest_close
    #         change = latest_close - prev_close
    #         change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
    #
    #         trend = "上涨📈" if change > 0 else "下跌📉" if change < 0 else "持平➖"
    #
    #         report += (
    #             f"- {data['stock_name']} ({stock_code}): "
    #             f"最新价 {latest_close:.2f} ({trend} {change_percent:.2f}%)\n"
    #         )
    #
    #     return report + f"\n图表路径: {chart_path}"
    #
    # # print("===", trend_analysis_tool("腾讯"))
    #
    #
    # # 3. 相关性分析工具
    # # def correlation_analysis_tool(stock_names: str) -> str:
    # #     """
    # #     分析多个股票之间的价格相关性
    # #     输入: 股票名称（多个用逗号分隔）
    # #     输出: 相关性分析报告和图表文件路径
    # #     """
    # #     stock_list = [name.strip() for name in stock_names.split(",")]
    # #
    # #     if not vector_db:
    # #         return "股票数据索引未加载，请先加载数据"
    # #
    # #     # 收集所有相关文档
    # #     stock_data = {}
    # #     for stock in stock_list:
    # #         if stock not in name_to_code:
    # #             continue
    # #
    # #         # 搜索该股票的所有文档
    # #         query = f"{stock} {name_to_code[stock]}"
    # #         docs = vector_db.similarity_search(query, k=10)
    # #
    # #         # 解析收盘价
    # #         closes = []
    # #         for doc in docs:
    # #             content = doc.page_content
    # #             kline_matches = re.findall(r":开\d+\.?\d*/收(\d+\.?\d*)，", content)
    # #             if kline_matches:
    # #                 closes.extend([float(price) for price in kline_matches])
    # #
    # #         if closes:
    # #             stock_data[stock] = closes
    # #
    # #     if len(stock_data) < 2:
    # #         return "需要至少两支股票进行分析"
    # #
    # #     # 创建相关性矩阵
    # #     min_length = min(len(prices) for prices in stock_data.values())
    # #     aligned_data = {stock: prices[-min_length:] for stock, prices in stock_data.items()}
    # #
    # #     df = pd.DataFrame(aligned_data)
    # #     corr_matrix = df.corr()
    # #
    # #     # 创建热力图
    # #     plt.figure(figsize=(10, 8))
    # #     plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    # #     plt.colorbar()
    # #     plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    # #     plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    # #     plt.title('股票价格相关性')
    # #
    # #     # 添加数值标签
    # #     for i in range(len(corr_matrix.columns)):
    # #         for j in range(len(corr_matrix.columns)):
    # #             plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
    # #                      ha="center", va="center", color="white")
    # #
    # #     # 保存图表
    # #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # #     chart_path = f"correlation_{timestamp}.png"
    # #     plt.savefig(chart_path)
    # #     plt.close()
    # #
    # #     # 生成分析报告
    # #     report = "📊 股票相关性分析报告\n\n"
    # #     report += f"分析股票: {', '.join(stock_list)}\n"
    # #     report += f"相关性图表已保存至: {chart_path}\n\n"
    # #
    # #     # 添加相关性摘要
    # #     report += "相关性摘要:\n"
    # #     for i, stock1 in enumerate(corr_matrix.columns):
    # #         for j, stock2 in enumerate(corr_matrix.columns):
    # #             if i < j:  # 避免重复和自身比较
    # #                 corr_value = corr_matrix.iloc[i, j]
    # #                 relation = "强正相关" if corr_value > 0.7 else \
    # #                     "中度正相关" if corr_value > 0.3 else \
    # #                         "弱正相关" if corr_value > 0 else \
    # #                             "弱负相关" if corr_value > -0.3 else \
    # #                                 "中度负相关" if corr_value > -0.7 else \
    # #                                     "强负相关"
    # #
    # #                 report += f"- {stock1} 与 {stock2}: {relation} ({corr_value:.2f})\n"
    # #
    # #     return report + f"\n图表路径: {chart_path}"
    #
    #
    # # 4. 基本面对比工具
    # # def fundamental_comparison_tool(stock_names: str) -> str:
    # #     """
    # #     对比多个股票的基本面数据
    # #     输入: 股票名称（多个用逗号分隔）
    # #     输出: 基本面对比报告
    # #     """
    # #     stock_list = [name.strip() for name in stock_names.split(",")]
    # #
    # #     if not vector_db:
    # #         return "股票数据索引未加载，请先加载数据"
    # #
    # #     # 收集股票基本面数据
    # #     fundamentals = []
    # #     for stock in stock_list:
    # #         if stock not in name_to_code:
    # #             continue
    # #
    # #         # 搜索该股票的最新文档
    # #         query = f"{stock} {name_to_code[stock]}"
    # #         docs = vector_db.similarity_search(query, k=1)
    # #         if not docs:
    # #             continue
    # #
    # #         doc = docs[0]
    # #         content = doc.page_content
    # #
    # #         # 从内容中提取基本面数据
    # #         fundamentals.append({
    # #             "name": stock,
    # #             "code": name_to_code[stock],
    # #             "content": content
    # #         })
    # #
    # #     if len(fundamentals) < 2:
    # #         return "需要至少两支股票进行对比"
    # #
    # #     # 创建对比报告
    # #     report = "📋 股票基本面对比报告\n\n"
    # #     report += f"对比股票: {', '.join(stock_list)}\n\n"
    # #
    # #     # 添加对比表格
    # #     report += "| 项目 | " + " | ".join([f["name"] for f in fundamentals]) + " |\n"
    # #     report += "|-----|" + "|".join(["-----" for _ in fundamentals]) + "|\n"
    # #
    # #     # 提取并对比关键指标
    # #     indicators = ["上市日期", "股票类型", "每手股数", "当前价", "涨跌幅", "成交量"]
    # #
    # #     for indicator in indicators:
    # #         row = f"| {indicator} | "
    # #         for f in fundamentals:
    # #             # 在内容中搜索指标值
    # #             match = re.search(f"{indicator}[：:]([^，。\n]+)", f["content"])
    # #             value = match.group(1).strip() if match else "N/A"
    # #             row += f"{value} | "
    # #         report += row + "\n"
    # #
    # #     report += "\n"
    # #
    # #     # 添加简要分析
    # #     report += "对比分析:\n"
    # #     # 这里可以添加更复杂的分析逻辑
    # #     report += "- 以上表格展示了各股票的关键基本面指标对比\n"
    # #     report += "- 请结合具体指标进行深入分析\n"
    # #
    # #     return report
    #
    #
    # # 5. 风险分析工具
    # def risk_analysis_tool(stock_name: str) -> str:
    #     """
    #     分析单支股票的风险指标
    #     输入: 股票名称
    #     输出: 风险分析报告
    #     """
    #     if stock_name not in name_to_code:
    #         return f"未识别的股票名称: {stock_name}"
    #
    #     if not vector_db:
    #         return "股票数据索引未加载，请先加载数据"
    #
    #     # 搜索该股票的最新文档
    #     query = f"{stock_name} {name_to_code[stock_name]}"
    #     docs = vector_db.similarity_search(query, k=5)
    #     if not docs:
    #         return f"未找到 {stock_name} 的相关数据"
    #
    #     # 分析所有相关文档
    #     price_changes = []
    #     volumes = []
    #     for doc in docs:
    #         content = doc.page_content
    #
    #         # 提取价格变化
    #         change_matches = re.findall(r"涨跌幅:(-?\d+\.?\d*)%", content)
    #         if change_matches:
    #             price_changes.extend([float(change) for change in change_matches])
    #
    #         # 提取成交量
    #         volume_matches = re.findall(r"成交量:([\d,]+)", content)
    #         if volume_matches:
    #             volumes.extend([int(vol.replace(",", "")) for vol in volume_matches])
    #
    #     # 计算风险指标
    #     report = f"⚠️ {stock_name} 风险分析报告\n\n"
    #
    #     if price_changes:
    #         volatility = np.std(price_changes)
    #         max_drop = min(price_changes)
    #         avg_change = np.mean(price_changes)
    #
    #         report += "📈 价格波动分析:\n"
    #         report += f"- 平均日涨跌幅: {avg_change:.2f}%\n"
    #         report += f"- 价格波动率: {volatility:.2f}%\n"
    #         report += f"- 最大单日跌幅: {max_drop:.2f}%\n"
    #
    #         if volatility > 5:
    #             report += "  → 高风险: 价格波动剧烈\n"
    #         elif volatility > 2:
    #             report += "  → 中风险: 价格波动适中\n"
    #         else:
    #             report += "  → 低风险: 价格波动平稳\n"
    #     else:
    #         report += "未找到价格波动数据\n"
    #
    #     if volumes:
    #         volume_volatility = np.std(volumes) / np.mean(volumes) * 100
    #         min_volume = min(volumes)
    #         avg_volume = np.mean(volumes)
    #
    #         report += "\n📊 成交量分析:\n"
    #         report += f"- 平均成交量: {avg_volume:,.0f}\n"
    #         report += f"- 成交量波动率: {volume_volatility:.2f}%\n"
    #         report += f"- 最低成交量: {min_volume:,.0f}\n"
    #
    #         if volume_volatility > 50:
    #             report += "  → 高风险: 成交量波动剧烈\n"
    #         elif volume_volatility > 30:
    #             report += "  → 中风险: 成交量波动明显\n"
    #         else:
    #             report += "  → 低风险: 成交量波动平稳\n"
    #     else:
    #         report += "未找到成交量数据\n"
    #
    #     # 添加总体风险评估
    #     report += "\n🔍 总体风险评估:\n"
    #
    #     risk_level = "低"
    #     if price_changes and volumes:
    #         if volatility > 5 or volume_volatility > 50:
    #             risk_level = "高"
    #         elif volatility > 3 or volume_volatility > 30:
    #             risk_level = "中"
    #
    #     report += f"- 风险等级: {risk_level}\n"
    #     report += "- 建议: "
    #
    #     if risk_level == "高":
    #         report += "此股票风险较高，投资需谨慎，建议小仓位或避免投资"
    #     elif risk_level == "中":
    #         report += "此股票风险适中，可考虑适量投资，但需密切关注市场变化"
    #     else:
    #         report += "此股票风险较低，适合稳健型投资者"
    #
    #     return report
    #
    #
    #
    #
    # # 6. 行业分析工具
    # # def sector_analysis_tool(sector: str) -> str:
    # #     """
    # #     分析特定行业的股票表现
    # #     输入: 行业名称（如"科技"、"金融"）
    # #     输出: 行业分析报告
    # #     """
    # #     if not vector_db:
    # #         return "股票数据索引未加载，请先加载数据"
    # #
    # #     # 搜索相关行业股票
    # #     results = vector_db.similarity_search(sector, k=10)
    # #
    # #     if not results:
    # #         return f"未找到 {sector} 行业的相关股票"
    # #
    # #     # 收集行业数据
    # #     sector_stocks = []
    # #     for doc in results:
    # #         content = doc.page_content
    # #         stock_name = doc.metadata.get("stock_name", "")
    # #
    # #         # 提取最新价格和涨跌幅
    # #         last_done_match = re.search(r"当前价:(\d+\.?\d*)", content)
    # #         change_match = re.search(r"涨跌幅:(-?\d+\.?\d*)%", content)
    # #
    # #         if last_done_match and change_match:
    # #             sector_stocks.append({
    # #                 "name": stock_name,
    # #                 "code": doc.metadata.get("stock_code", ""),
    # #                 "price": float(last_done_match.group(1)),
    # #                 "change": float(change_match.group(1))
    # #             })
    # #
    # #     if not sector_stocks:
    # #         return f"未找到 {sector} 行业股票的完整数据"
    # #
    # #     # 创建行业分析报告
    # #     report = f"🏭 {sector} 行业分析报告\n\n"
    # #     report += f"包含股票: {', '.join([s['name'] for s in sector_stocks])}\n\n"
    # #
    # #     # 行业平均表现
    # #     avg_change = np.mean([s["change"] for s in sector_stocks])
    # #     best_stock = max(sector_stocks, key=lambda x: x["change"])
    # #     worst_stock = min(sector_stocks, key=lambda x: x["change"])
    # #
    # #     report += "📊 行业整体表现:\n"
    # #     report += f"- 平均涨跌幅: {avg_change:.2f}%\n"
    # #     report += f"- 表现最佳: {best_stock['name']} ({best_stock['change']:.2f}%)\n"
    # #     report += f"- 表现最差: {worst_stock['name']} ({worst_stock['change']:.2f}%)\n\n"
    # #
    # #     # 创建表现图表
    # #     plt.figure(figsize=(10, 6))
    # #     names = [s["name"] for s in sector_stocks]
    # #     changes = [s["change"] for s in sector_stocks]
    # #
    # #     colors = ['green' if c >= 0 else 'red' for c in changes]
    # #     plt.bar(names, changes, color=colors)
    # #
    # #     plt.axhline(y=0, color='gray', linestyle='--')
    # #     plt.title(f"{sector} 行业股票涨跌幅")
    # #     plt.ylabel("涨跌幅 (%)")
    # #     plt.xticks(rotation=45)
    # #     plt.tight_layout()
    # #
    # #     # 保存图表
    # #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # #     chart_path = f"sector_{sector}_{timestamp}.png"
    # #     plt.savefig(chart_path)
    # #     plt.close()
    # #
    # #     report += f"行业表现图表已保存至: {chart_path}\n\n"
    # #
    # #     # 添加行业趋势分析
    # #     report += "🔍 行业趋势分析:\n"
    # #     if avg_change > 1:
    # #         report += f"- {sector} 行业近期表现强劲，整体上涨趋势明显\n"
    # #     elif avg_change < -1:
    # #         report += f"- {sector} 行业近期表现疲软，整体下跌趋势明显\n"
    # #     else:
    # #         report += f"- {sector} 行业近期表现平稳，无明显趋势\n"
    # #
    # #     report += f"- 行业内部分化: {best_stock['change'] - worst_stock['change']:.2f}个百分点\n"
    # #
    # #     return report + f"\n图表路径: {chart_path}"
    # # print("====", sector_analysis_tool("科技"))
    #
    # # 7. 数据更新工具
    # def update_stock_data_tool(stock_names: str) -> str:
    #     """
    #     更新指定股票的数据
    #     输入: 股票名称（多个用逗号分隔）
    #     输出: 更新结果报告
    #     """
    #     stock_list = [name.strip() for name in stock_names.split(",")]
    #
    #     # 这里需要实现实际的数据更新逻辑
    #     # 在实际应用中，这里会调用之前的 add_or_update_stock_data 函数
    #
    #     # 模拟更新过程
    #     results = []
    #     for stock in stock_list:
    #         if stock in name_to_code:
    #             results.append(f"{stock} 数据已更新")
    #         else:
    #             results.append(f"{stock} 未识别")
    #
    #     return "更新结果:\n- " + "\n- ".join(results)
    #
    # print("====", update_stock_data_tool("腾讯"))
    #
    #
    # # 创建工具列表
    # tools = [
    #     Tool(
    #         name="股票搜索",
    #         func=stock_search_tool,
    #         description="搜索股票数据，输入查询内容，返回相关股票信息"
    #     ),
    #     Tool(
    #         name="趋势分析",
    #         func=trend_analysis_tool,
    #         description="分析股票价格趋势，输入股票名称（多个用逗号分隔）和天数（可选，默认5天），返回趋势报告和图表"
    #     ),
    #     # Tool(
    #     #     name="相关性分析",
    #     #     func=correlation_analysis_tool,
    #     #     description="分析多个股票之间的价格相关性，输入股票名称（多个用逗号分隔），返回相关性报告和图表"
    #     # ),
    #     # Tool(
    #     #     name="基本面对比",
    #     #     func=fundamental_comparison_tool,
    #     #     description="对比多个股票的基本面数据，输入股票名称（多个用逗号分隔），返回对比报告"
    #     # ),
    #     Tool(
    #         name="风险分析",
    #         func=risk_analysis_tool,
    #         description="分析单支股票的风险指标，输入股票名称，返回风险报告"
    #     ),
    #     # Tool(
    #     #     name="行业分析",
    #     #     func=sector_analysis_tool,
    #     #     description="分析特定行业的股票表现，输入行业名称（如'科技'、'金融'），返回行业报告"
    #     # ),
    #     Tool(
    #         name="数据更新",
    #         func=update_stock_data_tool,
    #         description="更新指定股票的数据，输入股票名称（多个用逗号分隔），返回更新结果"
    #     )
    # ]
    #
    #
    # # 创建 Agent
    # def create_agent():
    #     """创建并返回股票分析 Agent"""
    #     # 系统提示词
    #     system_prompt = """你是一个专业的股票数据分析师，擅长使用各种工具分析港股市场数据。
    #     你可以使用以下工具来分析FAISS数据库中存储的股票数据：
    #     - 股票搜索：搜索相关股票信息
    #     - 趋势分析：分析股票价格趋势
    #     - 相关性分析：分析多个股票之间的价格相关性
    #     - 基本面对比：对比多个股票的基本面数据
    #     - 风险分析：分析单支股票的风险指标
    #     - 行业分析：分析特定行业的股票表现
    #     - 数据更新：更新指定股票的数据
    #
    #     请根据用户的问题选择最合适的工具进行分析。如果用户的问题需要多个步骤，可以连续使用多个工具。
    #
    #     当使用图表生成工具时，请将图表路径包含在回复中，以便用户查看。
    #
    #     你的分析应该专业、全面，并提供有价值的投资见解。
    #     """
    #
    #     # 创建提示模板
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", system_prompt),
    #         MessagesPlaceholder("chat_history", optional=True),
    #         ("human", "{input}"),
    #         MessagesPlaceholder("agent_scratchpad")
    #     ])
    #
    #     # 选择LLM模型
    #     api_key = os.getenv("DEEPSEEK_API_KEY")
    #     # 封装 OpenAI 或兼容 API 成为一个 LangChain 可用的语言模型接口（LLM）
    #     llm = ChatOpenAI(
    #         api_key=api_key,
    #         base_url="https://api.deepseek.com",
    #         model="deepseek-chat",
    #         temperature=0.7  # 控制语言模型输出内容 随机性 的参数，官方推荐0.7
    #     )
    #
    #     # 创建Agent
    #     agent = create_openai_functions_agent(llm, tools, prompt)
    #
    #     # 创建执行器
    #     agent_executor = AgentExecutor(
    #         agent=agent,
    #         tools=tools,
    #         verbose=True,
    #         return_intermediate_steps=True
    #     )
    #
    #     return agent_executor
    #
    #
    # # 创建Agent实例
    # stock_analyst_agent = create_agent()
    #
    #
    # # 主函数 - 用户交互
    # def main():
    #     print("欢迎使用港股数据分析助手！输入'退出'结束对话。")
    #     print("支持的分析类型：趋势分析、相关性分析、基本面对比、风险评估、行业分析等\n")
    #
    #     while True:
    #         user_input = input("\n您想分析什么？\n> ")
    #
    #         if user_input.lower() in ["退出", "exit", "quit"]:
    #             print("感谢使用，再见！")
    #             break
    #
    #         try:
    #             # 执行Agent
    #             result = stock_analyst_agent.invoke({"input": user_input})
    #
    #             # 显示结果
    #             print("\n" + "=" * 60)
    #             print("📊 分析结果:")
    #             print(result["output"])
    #             print("=" * 60)
    #
    #             # 显示中间步骤（可选）
    #             if result.get("intermediate_steps"):
    #                 print("\n🔧 分析步骤:")
    #                 for step in result["intermediate_steps"]:
    #                     action = step[0]
    #                     observation = step[1]
    #                     print(f"- 操作: {action.tool}（输入: {action.tool_input}）")
    #                     print(f"  结果: {observation[:200]}...")
    #
    #         except Exception as e:
    #             print(f"分析过程中出错: {str(e)}")
    #
    #
    # if __name__ == "__main__":
    #     main()
    embedder = pipeline(
        "feature-extraction",
        model=MODEL_PATH,
        device=0 if torch.cuda.is_available() else -1,  # 0 表示第一个 GPU
        local_files_only=True
    )

    # 获取嵌入
    result = embedder("测试文本")
    embeddings = result[0][0]  # 获取第一个 token 的嵌入（通常是 [CLS] token）

    print(f"✅ 模型加载成功！嵌入维度: {len(embeddings)}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")