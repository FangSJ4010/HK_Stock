import os
from longbridge.openapi import QuoteContext, Config, SubType, Period, AdjustType, CalcIndex
from langchain.tools import Tool
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

config = Config(
    app_key = os.getenv("LONGPORT_APP_KEY"),
    app_secret = os.getenv("LONGPORT_APP_SECRET"),
    access_token = os.getenv("LONGPORT_ACCESS_TOKEN")
)



'''
指定代码，港股数据获取
# 获取个人账户信息，包括账户现金余额、净资产、可融金额等
# ctx = TradeContext(config)
# resp = ctx.account_balance()
# print(resp)
'''
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

# 封装数据查询函数
def hk_query_tool_func(stock_name: str):
    # 股票名称到代码映射
    name_to_code = {
        "腾讯": "00700.HK",
        "阿里巴巴": "09988.HK",
        "美团": "03690.HK",
        "小米": "01810.HK",
        "粤港湾": "01396.HK",
    }

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
            res += f"{point.timestamp}: 价格={point.price}，成交量={point.volume}\n"
    else:
        res += "无分时数据\n"
    res += "\n"

    # 历史K线（展示最近5天）
    res += f"【最近 5 日K线】\n"
    if klines:
        for k in klines[-5:]:
            res += f"{k.timestamp.date()} 开盘={k.open} 收盘={k.close} 最高={k.high} 最低={k.low} 成交量={k.volume}\n"
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


if __name__ == "__main__":
    stock_data = hk_code_data("01396.HK")



