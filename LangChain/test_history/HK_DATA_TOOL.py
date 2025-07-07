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
        # 打印调试
        # if not quote:
        #     print(f"未能获取 {stock_code} 的实时行情数据，可能未发生交易")
        # else:
        #     q = quote[0]
        #     print(f"\n===== {stock_code} 实时行情 =====")
        #     print(f"最新价: {q.last_done}")
        #     # 计算涨跌幅
        #     if q.prev_close is not None and q.last_done is not None:
        #         change_rate = (q.last_done - q.prev_close) / q.prev_close
        #         print(f"涨跌幅: {change_rate:.2%}")
        #     else:
        #         print("涨跌幅: 无法计算")
        #     print(f"成交量: {q.volume}")


        '''获取分时数据'''
        intraday = ctx.intraday(stock_code)
        # print(f"\n===== {stock_code} 当日分时 =====")
        # for tick in intraday[-5:]:
        #     print(f"{tick.timestamp}: 价格={tick.price} 成交量={tick.volume}")

        '''获取历史K线'''
        klines = ctx.candlesticks(
            stock_code,
            period=Period.Day,
            count=5,
            adjust_type=AdjustType.NoAdjust
        )
        # print(f"\n===== {stock_code} 历史K线(日) =====")
        # for k in klines:
        #     date_str = k.timestamp.strftime('%Y-%m-%d')  # 正确格式化时间
        #     print(f"{date_str}: 开={k.open} 高={k.high} 低={k.low} 收={k.close} 量={k.volume}")

        '''获取公司基本信息'''
        info = infos[0]
        # print(f"\n===== {stock_code} 基本信息 =====")
        # print(f"公司名称: {info.name_cn}")
        # print(f"英文名称: {info.name_en}")
        # print(f"每手股数: {info.lot_size}")
        # print(f"交易所: {info.exchange}")


        '''获取计算指标'''
        indicators = ctx.calc_indexes([stock_code],
                                      [
                                       CalcIndex.LastDone,  # 最新价
                                       CalcIndex.ChangeRate,  # 涨跌幅
                                       CalcIndex.ChangeValue,  # 涨跌额
                                       CalcIndex.Volume,  # 成交量
                                       CalcIndex.Turnover,  # 成交额
                                       ])
        print("=====",indicators)


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

# 用于封装你的数据查询函数
def hk_query_tool_func(stock_name: str):
    # 加入映射
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

    # 构造回答
    quote = data["quote"]
    info = data["info"]

    res = f"{info.name_cn}（{stock_code}）最新行情：\n"
    if quote:
        res += f"当前价：{quote.last_done} HKD\n"
        # 计算涨跌幅
        if quote.prev_close is not None and quote.last_done is not None:
            change_rate = (quote.last_done - quote.prev_close) / quote.prev_close
            res += f"涨跌幅：{change_rate:.2%}\n"
        else:
            res += f"涨跌幅无法计算\n"

        res += f"成交量：{quote.volume}\n"
    else:
        res += f"暂无实时行情数据（可能未开盘）\n"

    return res

# 封装成 LangChain Tool
hk_tool = Tool(
    name="港股信息查询工具",
    func=hk_query_tool_func,
    description="输入港股公司名称（如'腾讯'），获取其实时行情和基本信息"
)


if __name__ == "__main__":
    stock_data = hk_code_data("01396.HK")



