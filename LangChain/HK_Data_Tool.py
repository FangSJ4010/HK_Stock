import os
from longbridge.openapi import QuoteContext, Config, Period, AdjustType, CalcIndex
from langchain.tools import Tool
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from datetime import datetime
# from RAG_Data.Embedding_Vector import Creat_Embeddings


# 加载 .env 文件
load_dotenv()

# 配置路径

FAISS_INDEX_PATH = "E:/Pycharm/LangChain/FAISS_Stock_Data"
INDEX_NAME = "hk_stock_index"

# 创建自定义嵌入对象
embedding = Creat_Embeddings()

# 初始化Longbridge配置
config = Config(
    app_key=os.getenv("LONGPORT_APP_KEY"),
    app_secret=os.getenv("LONGPORT_APP_SECRET"),
    access_token=os.getenv("LONGPORT_ACCESS_TOKEN")
)

# 股票名称到代码映射
name_to_code = {
    "腾讯": "00700.HK",
    "阿里巴巴": "09988.HK",
    "美团": "03690.HK",
    "小米": "01810.HK",
    "粤港湾": "01396.HK",
}


# 获取股票数据
def hk_code_data(stock_code: str):
    try:
        ctx = QuoteContext(config)

        # 获取股票静态信息
        infos = ctx.static_info([stock_code])
        if not infos:
            print(f"未找到股票代码: {stock_code}")
            return None

        '''获取实时行情'''
        quote = ctx.quote([stock_code])

        '''获取分时数据'''
        intraday = ctx.intraday(stock_code)

        '''获取历史K线'''
        klines = ctx.candlesticks(
            stock_code,
            period=Period.Day,
            count=5,
            adjust_type=AdjustType.NoAdjust
        )

        '''获取公司基本信息'''
        info = infos[0]

        '''获取计算指标'''
        indicators = ctx.calc_indexes([stock_code],
                                      [
                                          CalcIndex.LastDone,  # 最新价
                                          CalcIndex.ChangeRate,  # 涨跌幅
                                          CalcIndex.ChangeValue,  # 涨跌额
                                          CalcIndex.Volume,  # 成交量
                                          CalcIndex.Turnover,  # 成交额
                                      ])

        return {
            "quote": quote[0] if quote else None,
            "intraday": intraday,
            "klines": klines,
            "info": info,
            "indicators": indicators
        }

    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        return None


# 将股票数据转换为文本用于嵌入
def stock_data_to_text(data):
    """将股票数据转换为适合嵌入的文本格式"""
    quote = data["quote"]
    info = data["info"]
    klines = data["klines"]
    intraday = data["intraday"]
    indicators = data["indicators"]

    # 基本信息
    text = f"{getattr(info, 'name_cn', '')} ({getattr(info, 'symbol', '')}) "
    text += f"上市于{getattr(info, 'listing_date', '')}，类型:{getattr(info, 'stock_type', '')}。"

    # 实时行情
    if quote:
        text += f"当前价:{quote.last_done}，昨收价:{quote.prev_close}，"
        if quote.prev_close and quote.last_done:
            change_rate = (quote.last_done - quote.prev_close) / quote.prev_close
            text += f"涨跌幅:{change_rate:.2%}，"
        text += f"成交量:{quote.volume}，成交额:{quote.turnover}。"

    # 分时数据（只取最近3条）
    if intraday:
        text += "最近分时:"
        for point in intraday[-3:]:
            text += f"{point.timestamp.strftime('%H:%M')}:{point.price}，"

    # 历史K线（只取最近3天）
    if klines:
        text += "最近K线:"
        for k in klines[-3:]:
            text += f"{k.timestamp.date()}:开{k.open}/收{k.close}，"

    # 计算指标
    if indicators:
        text += "指标:"
        for idx in indicators:
            for field in ["last_done", "change_rate", "volume", "turnover"]:
                value = getattr(idx, field, None)
                if isinstance(value, tuple) and value[0] == "Some":
                    value = value[1]
                if value is not None:
                    display_name = {
                        "last_done": "最新价",
                        "change_rate": "涨跌幅",
                        "volume": "成交量",
                        "turnover": "成交额",
                    }.get(field, field)
                    text += f"{display_name}:{value}，"

    return text.strip('，')  # 移除末尾多余的逗号


# 添加或更新向量库中的股票数据
def add_or_update_stock_data(stock_name: str):
    """添加或更新股票数据到向量库"""
    if stock_name not in name_to_code:
        return f"未识别的股票名称：{stock_name}"

    stock_code = name_to_code[stock_name]
    data = hk_code_data(stock_code)

    if not data:
        return f"{stock_name}（{stock_code}）的数据获取失败"

    # 转换为嵌入文本
    content = stock_data_to_text(data)

    # 创建文档对象，包含元数据
    metadata = {
        "stock_name": stock_name,
        "stock_code": stock_code,
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    doc = Document(page_content=content, metadata=metadata)

    # 检查索引是否存在
    if os.path.exists(os.path.join(FAISS_INDEX_PATH, f"{INDEX_NAME}.faiss")):
        # 加载已有索引
        db = FAISS.load_local(
            folder_path=FAISS_INDEX_PATH,
            embeddings=embedding,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True
        )

        print(f"已加载现有索引 '{INDEX_NAME}'，包含 {db.index.ntotal} 个文档")

        # 检查该股票是否已存在
        existing_ids = []
        for doc_id, existing_doc in db.docstore._dict.items():
            if existing_doc.metadata.get("stock_code") == stock_code:
                existing_ids.append(doc_id)

        # 删除已有的记录
        if existing_ids:
            db.delete(existing_ids)
            print(f"已删除 {len(existing_ids)} 条旧记录")

        # 添加新文档
        db.add_documents([doc])
        print(f"已添加 {stock_name} 的新数据")
    else:
        # 创建新索引
        db = FAISS.from_documents([doc], embedding)
        print(f"创建新索引 '{INDEX_NAME}'，添加 {stock_name} 的数据")

    # 保存更新后的索引
    db.save_local(FAISS_INDEX_PATH, index_name=INDEX_NAME)
    print(f"索引已保存到: {FAISS_INDEX_PATH}")

    return f"{stock_name} 数据已更新"


# 封装数据查询函数
def hk_query_tool_func(stock_name: str):
    """港股信息查询工具的主函数"""
    if stock_name not in name_to_code:
        return f"未识别的股票名称：{stock_name}，请提供正确的港股名称（如：腾讯、美团）"

    stock_code = name_to_code[stock_name]
    data = hk_code_data(stock_code)

    if not data:
        return f"{stock_name}（{stock_code}）的数据获取失败"

    quote = data["quote"]
    info = data["info"]
    klines = data["klines"]
    intraday = data["intraday"]
    indicators = data["indicators"]

    res = f"📈 {getattr(info, 'name_cn', stock_name)}（{stock_code}）详细信息：\n\n"

    # 公司基本信息
    res += f"【公司基本信息】\n"
    res += f"中文名称：{getattr(info, 'name_cn', 'N/A')}\n"
    res += f"英文名称：{getattr(info, 'name_en', 'N/A')}\n"
    res += f"上市日期：{getattr(info, 'listing_date', 'N/A')}\n"
    res += f"交易所：{getattr(info, 'exchange_code', 'N/A')}\n"
    res += f"股票类型：{getattr(info, 'stock_type', 'N/A')}\n"
    res += f"每手股数：{getattr(info, 'lot_size', 'N/A')} 股\n\n"

    # 实时行情
    if quote:
        res += f"【实时行情】\n"
        res += f"当前价：{quote.last_done} HKD\n"
        res += f"昨收价：{quote.prev_close} HKD\n"
        if quote.prev_close and quote.last_done:
            change_rate = (quote.last_done - quote.prev_close) / quote.prev_close
            res += f"涨跌幅：{change_rate:.2%}\n"
        res += f"成交量：{quote.volume}\n"
        res += f"成交额：{quote.turnover}\n\n"
    else:
        res += "【实时行情】暂无数据（可能未开盘）\n\n"

    # 分时数据（展示最后5条）
    res += f"【分时数据】（最近 5 条）\n"
    if intraday:
        for point in intraday[-5:]:
            timestamp_str = point.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            res += f"{timestamp_str}: 价格={point.price}，成交量={point.volume}\n"
    else:
        res += "无分时数据\n"
    res += "\n"

    # 历史K线（展示最近5天）
    res += f"【最近 5 日K线】\n"
    if klines:
        for k in klines[-5:]:
            date_str = k.timestamp.strftime("%Y-%m-%d")
            res += f"{date_str} 开盘={k.open} 收盘={k.close} 最高={k.high} 最低={k.low} 成交量={k.volume}\n"
    else:
        res += "无K线数据\n"
    res += "\n"

    res += "【计算指标】\n"
    if indicators:
        for idx in indicators:
            for field in ["last_done", "change_rate", "volume", "turnover"]:
                value = getattr(idx, field, None)
                if isinstance(value, tuple) and value[0] == "Some":
                    value = value[1]
                if value is not None:
                    display_name = {
                        "last_done": "最新价",
                        "change_rate": "涨跌幅",
                        "volume": "成交量",
                        "turnover": "成交额",
                    }.get(field, field)
                    res += f"{display_name}: {value}\n"
    else:
        res += "无计算指标\n"

    return res


# 封装成 LangChain Tool
hk_tool = Tool(
    name="港股信息查询工具",
    func=hk_query_tool_func,
    description="输入港股公司名称（如'腾讯'），获取其实时行情、分时数据、K线、指标和公司信息"
)

# 股票数据向量化工具
stock_vector_tool = Tool(
    name="港股数据向量化",
    func=add_or_update_stock_data,
    description="输入港股公司名称（如'腾讯'），将其数据向量化并存储到FAISS数据库"
)


# 搜索股票数据
def search_stock_data(query: str, k: int = 3):
    """在向量库中搜索相关股票数据"""
    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, f"{INDEX_NAME}.faiss")):
        return "尚未创建股票数据索引，请先使用港股数据向量化工具添加数据"

    # 加载索引
    db = FAISS.load_local(
        folder_path=FAISS_INDEX_PATH,
        embeddings=embedding,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True
    )

    # 执行搜索
    results = db.similarity_search(query, k=k)

    # 格式化结果
    res = f"🔍 搜索 '{query}' 的结果（显示前 {k} 条）:\n\n"
    for i, doc in enumerate(results):
        stock_name = doc.metadata.get("stock_name", "未知股票")
        stock_code = doc.metadata.get("stock_code", "未知代码")
        update_time = doc.metadata.get("update_time", "未知时间")

        res += f"🏷️ {i + 1}. {stock_name} ({stock_code})\n"
        res += f"🕒 更新时间: {update_time}\n"
        res += f"📝 摘要: {doc.page_content[:100]}...\n\n"

    return res


# 股票数据搜索工具
stock_search_tool = Tool(
    name="港股数据搜索",
    func=search_stock_data,
    description="输入查询内容，在向量库中搜索相关股票数据"
)

if __name__ == "__main__":
    # 示例：添加或更新股票数据
    print(add_or_update_stock_data("腾讯"))

    # 示例：搜索股票数据
    print(search_stock_data("科技公司", k=2))

    # 示例：查询详细数据
    print(hk_query_tool_func("腾讯"))