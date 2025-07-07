from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
import matplotlib.pyplot as plt
import textwrap
import re
import os
from longbridge.openapi import Config
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS
import numpy as np
from datetime import datetime
from RAG_Data.Embedding_model import Creat_Embeddings
from dotenv import load_dotenv

load_dotenv()

# 初始化Longbridge配置
config = Config(
    app_key=os.getenv("LONGPORT_APP_KEY"),
    app_secret=os.getenv("LONGPORT_APP_SECRET"),
    access_token=os.getenv("LONGPORT_ACCESS_TOKEN")
)

# 配置路径
MODEL_PATH = "E:/Pycharm/LangChain/embedding"
FAISS_INDEX_PATH = "E:/Pycharm/LangChain/FAISS_Stock_Data"
INDEX_NAME = "hk_stock_index"


# 创建自定义嵌入对象
embedding = Creat_Embeddings()

# 股票名称到代码映射
name_to_code = {
    "腾讯": "00700.HK",
    "阿里巴巴": "09988.HK",
    "美团": "03690.HK",
    "小米": "01810.HK",
    "粤港湾": "01396.HK",
    "京东": "09618.HK",
    "百度": "09888.HK",
    "网易": "09999.HK",
    "比亚迪": "01211.HK",
    "中国平安": "02318.HK",
}


# 加载 FAISS 向量库
def load_faiss_index():
    """加载 FAISS 向量库"""
    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, f"{INDEX_NAME}.faiss")):
        print(f"未存储{INDEX_NAME}相关数据")
        return None

    try:
        db = FAISS.load_local(FAISS_INDEX_PATH, embedding, index_name=INDEX_NAME, allow_dangerous_deserialization=True)
        print(f"已加载股票数据索引，包含 {db.index.ntotal} 个文档")
        return db

    except Exception as e:
        print(f"加载索引失败: {str(e)}")
        return None

# 初始化向量库
vector_db = load_faiss_index()



# 1. 基础搜索工具
def stock_search_tool(query: str, k: int = 5) -> str:
    """在向量库中搜索相关股票数据"""
    if not vector_db:
        return "股票数据索引未加载，请先加载数据"

    results = vector_db.similarity_search(query, k=k)

    output = []
    for i, doc in enumerate(results):
        stock_name = doc.metadata.get("stock_name", "未知股票")
        stock_code = doc.metadata.get("stock_code", "未知代码")
        update_time = doc.metadata.get("update_time", "未知时间")

        # 格式化文本摘要
        content = textwrap.fill(doc.page_content, width=80)

        output.append(
            f"🔍 {i + 1}. {stock_name} ({stock_code})\n"
            f"🕒 更新时间: {update_time}\n"
            f"📝 摘要: {content}\n"
            f"{'-' * 40}"
        )

    return "\n".join(output)


# 2. 趋势分析工具
def trend_analysis_tool(stock_names: str, days: int = 5) -> str:
    """
    分析股票趋势，返回文本报告和图表路径
    输入: 股票名称（多个用逗号分隔），分析天数
    输出: 趋势分析报告和图表文件路径
    """
    # 1. 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
    plt.rcParams['axes.unicode_minus'] = False

    stock_list = [name.strip() for name in stock_names.split(",")]

    if not vector_db:
        return "股票数据索引未加载，请先加载数据"

    # 收集所有相关文档
    all_docs = []
    for stock in stock_list:
        if stock not in name_to_code:
            continue

        # 搜索该股票的所有文档
        query = f"{stock} {name_to_code[stock]}"
        docs = vector_db.similarity_search(query, k=10)
        all_docs.extend(docs)

    if not all_docs:
        return "未找到相关股票数据"

    # 解析K线数据
    kline_data = {}
    for doc in all_docs:
        stock_code = doc.metadata.get("stock_code")
        if not stock_code:
            continue

        # 从文档内容中提取K线数据
        content = doc.page_content
        kline_matches = re.findall(r"最近K线:([^:]+):开(\d+\.?\d*)/收(\d+\.?\d*)，", content)

        if not kline_matches:
            continue

        # 组织K线数据
        if stock_code not in kline_data:
            kline_data[stock_code] = {
                "dates": [],
                "opens": [],
                "closes": [],
                "stock_name": doc.metadata.get("stock_name", stock_code)
            }

        for match in kline_matches:
            date_str, open_price, close_price = match
            kline_data[stock_code]["dates"].append(date_str)
            kline_data[stock_code]["opens"].append(float(open_price))
            kline_data[stock_code]["closes"].append(float(close_price))

    if not kline_data:
        return "未找到K线数据"

    # 创建趋势图表
    plt.figure(figsize=(12, 8))

    for stock_code, data in kline_data.items():
        # 只取最近days天的数据
        dates = data["dates"][-days:]
        closes = data["closes"][-days:]

        if len(dates) < 2:
            continue

        # 计算移动平均
        moving_avg = np.convolve(closes, np.ones(3) / 3, mode='valid')

        plt.plot(dates, closes, 'o-', label=f"{data['stock_name']} ({stock_code})")
        plt.plot(dates[1:-1], moving_avg, '--', alpha=0.5)

    plt.title(f"股票价格趋势 ({days}天)")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = f"stock_trend_{timestamp}.png"
    plt.savefig(chart_path)
    plt.close()

    # 生成分析报告
    report = f"📈 股票趋势分析报告 ({days}天)\n\n"
    report += f"分析股票: {', '.join(stock_list)}\n"
    report += f"趋势图表已保存至: {chart_path}\n\n"

    # 添加简要分析
    report += "分析摘要:\n"
    for stock_code, data in kline_data.items():
        if len(data["closes"]) < 2:
            continue

        latest_close = data["closes"][-1]
        prev_close = data["closes"][-2] if len(data["closes"]) > 1 else latest_close
        change = latest_close - prev_close
        change_percent = (change / prev_close) * 100 if prev_close != 0 else 0

        trend = "上涨📈" if change > 0 else "下跌📉" if change < 0 else "持平➖"

        report += (
            f"- {data['stock_name']} ({stock_code}): "
            f"最新价 {latest_close:.2f} ({trend} {change_percent:.2f}%)\n"
        )

    return report + f"\n图表路径: {chart_path}"


# 5. 风险分析工具
def risk_analysis_tool(stock_name: str) -> str:
    """
    分析单支股票的风险指标
    输入: 股票名称
    输出: 风险分析报告
    """
    if stock_name not in name_to_code:
        return f"未识别的股票名称: {stock_name}"

    if not vector_db:
        return "股票数据索引未加载，请先加载数据"

    # 搜索该股票的最新文档
    query = f"{stock_name} {name_to_code[stock_name]}"
    docs = vector_db.similarity_search(query, k=5)
    if not docs:
        return f"未找到 {stock_name} 的相关数据"

    # 分析所有相关文档
    price_changes = []
    volumes = []
    for doc in docs:
        content = doc.page_content

        # 提取价格变化
        change_matches = re.findall(r"涨跌幅:(-?\d+\.?\d*)%", content)
        if change_matches:
            price_changes.extend([float(change) for change in change_matches])

        # 提取成交量
        volume_matches = re.findall(r"成交量:([\d,]+)", content)
        if volume_matches:
            volumes.extend([int(vol.replace(",", "")) for vol in volume_matches])

    # 计算风险指标
    report = f"⚠️ {stock_name} 风险分析报告\n\n"

    if price_changes:
        volatility = np.std(price_changes)
        max_drop = min(price_changes)
        avg_change = np.mean(price_changes)

        report += "📈 价格波动分析:\n"
        report += f"- 平均日涨跌幅: {avg_change:.2f}%\n"
        report += f"- 价格波动率: {volatility:.2f}%\n"
        report += f"- 最大单日跌幅: {max_drop:.2f}%\n"

        if volatility > 5:
            report += "  → 高风险: 价格波动剧烈\n"
        elif volatility > 2:
            report += "  → 中风险: 价格波动适中\n"
        else:
            report += "  → 低风险: 价格波动平稳\n"
    else:
        report += "未找到价格波动数据\n"

    if volumes:
        volume_volatility = np.std(volumes) / np.mean(volumes) * 100
        min_volume = min(volumes)
        avg_volume = np.mean(volumes)

        report += "\n📊 成交量分析:\n"
        report += f"- 平均成交量: {avg_volume:,.0f}\n"
        report += f"- 成交量波动率: {volume_volatility:.2f}%\n"
        report += f"- 最低成交量: {min_volume:,.0f}\n"

        if volume_volatility > 50:
            report += "  → 高风险: 成交量波动剧烈\n"
        elif volume_volatility > 30:
            report += "  → 中风险: 成交量波动明显\n"
        else:
            report += "  → 低风险: 成交量波动平稳\n"
    else:
        report += "未找到成交量数据\n"

    # 添加总体风险评估
    report += "\n🔍 总体风险评估:\n"

    risk_level = "低"
    if price_changes and volumes:
        if volatility > 5 or volume_volatility > 50:
            risk_level = "高"
        elif volatility > 3 or volume_volatility > 30:
            risk_level = "中"

    report += f"- 风险等级: {risk_level}\n"
    report += "- 建议: "

    if risk_level == "高":
        report += "此股票风险较高，投资需谨慎，建议小仓位或避免投资"
    elif risk_level == "中":
        report += "此股票风险适中，可考虑适量投资，但需密切关注市场变化"
    else:
        report += "此股票风险较低，适合稳健型投资者"

    return report


# 7. 数据更新工具
def update_stock_data_tool(stock_names: str) -> str:
    """
    更新指定股票的数据
    输入: 股票名称（多个用逗号分隔）
    输出: 更新结果报告
    """
    stock_list = [name.strip() for name in stock_names.split(",")]

    # 这里需要实现实际的数据更新逻辑
    # 在实际应用中，这里会调用之前的 add_or_update_stock_data 函数

    # 模拟更新过程
    results = []
    for stock in stock_list:
        if stock in name_to_code:
            results.append(f"{stock} 数据已更新")
        else:
            results.append(f"{stock} 未识别")

    return "更新结果:\n- " + "\n- ".join(results)


# 创建工具列表
Analyse_Tools = [
    Tool(
        name="股票搜索",
        func=stock_search_tool,
        description="搜索股票数据，输入查询内容，返回相关股票信息"
    ),
    Tool(
        name="趋势分析",
        func=trend_analysis_tool,
        description="分析股票价格趋势，输入股票名称（多个用逗号分隔）和天数（可选，默认5天），返回趋势报告和图表"
    ),
    Tool(
        name="风险分析",
        func=risk_analysis_tool,
        description="分析单支股票的风险指标，输入股票名称，返回风险报告"
    ),
    Tool(
        name="数据更新",
        func=update_stock_data_tool,
        description="更新指定股票的数据，输入股票名称（多个用逗号分隔），返回更新结果"
    )
]


# 创建 Agent
def create_agent():
    """创建并返回股票分析 Agent"""
    # 系统提示词
    system_prompt = """你是一个专业的股票数据分析师，擅长使用各种工具分析港股市场数据。
    你可以使用以下工具来分析FAISS数据库中存储的股票数据：
    - 股票搜索：搜索相关股票信息
    - 趋势分析：分析股票价格趋势
    - 相关性分析：分析多个股票之间的价格相关性
    - 基本面对比：对比多个股票的基本面数据
    - 风险分析：分析单支股票的风险指标
    - 行业分析：分析特定行业的股票表现
    - 数据更新：更新指定股票的数据

    请根据用户的问题选择最合适的工具进行分析。如果用户的问题需要多个步骤，可以连续使用多个工具。

    当使用图表生成工具时，请将图表路径包含在回复中，以便用户查看。

    你的分析应该专业、全面，并提供有价值的投资见解。
    """

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    # 选择LLM模型
    api_key = os.getenv("DEEPSEEK_API_KEY")
    # 封装 OpenAI 或兼容 API 成为一个 LangChain 可用的语言模型接口（LLM）
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0.7  # 控制语言模型输出内容 随机性 的参数，官方推荐0.7
    )

    # 创建Agent
    agent = create_openai_functions_agent(llm, Analyse_Tools, prompt)

    # 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=Analyse_Tools,
        verbose=True,
        return_intermediate_steps=True
    )

    return agent_executor


# 创建Agent实例
stock_analyst_agent = create_agent()

api_key = os.getenv("DEEPSEEK_API_KEY")
# 封装 OpenAI 或兼容 API 成为一个 LangChain 可用的语言模型接口（LLM）
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7  # 控制语言模型输出内容 随机性 的参数，官方推荐0.7
)


# response = get_hk_data_Agent.run("帮我查一下腾讯的详细信息，并分析一下形势")
# print(response)
