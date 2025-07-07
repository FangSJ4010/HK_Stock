import os
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import ScrapeWebsiteTool
from langchain import hub
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

# 设置 DeepSeek API 配置
os.environ["DEEPSEEK_API_KEY"] = "sk-8075b254effb4d14bf2c21563d564322"  # 替换为你的实际密钥
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
os.environ["SERPER_API_KEY"] = "6780cf9c2f59a392c9aa8cebee07f05fc3a4e06d"  # 替换为你的实际密钥

# 初始化 DeepSeek LLM
deepseek_llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_base="https://api.deepseek.com/v1",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3
)


# 1. 定义工具类
class FinancialDataTool(BaseTool):
    name = "financial_data_fetcher"
    description = "获取股票财务数据，包括历史价格、成交量和技术指标"

    def _run(self, symbol: str, period: str = "1y", indicators: list = ["SMA", "RSI"]) -> dict:
        """获取股票财务数据"""
        try:
            # 使用 yfinance 获取股票数据
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)

            # 计算技术指标
            if "SMA" in indicators:
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['SMA_200'] = hist['Close'].rolling(window=200).mean()

            if "RSI" in indicators:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                hist['RSI'] = 100 - (100 / (1 + rs))

            # 转换为字典格式
            data = hist.reset_index().to_dict(orient='records')

            # 添加基本信息
            info = stock.info
            basic_info = {
                'symbol': symbol,
                'name': info.get('shortName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'marketCap': info.get('marketCap', 0),
                'peRatio': info.get('trailingPE', 0),
                'dividendYield': info.get('dividendYield', 0)
            }

            return {
                'basic_info': basic_info,
                'historical_data': data,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}


class NewsAnalysisTool(BaseTool):
    name = "financial_news_analyzer"
    description = "获取并分析特定股票的金融新闻和情绪"

    def _run(self, symbol: str, num_articles: int = 5) -> dict:
        """获取和分析股票相关新闻"""
        try:
            # 使用 Serper API 获取新闻
            url = "https://google.serper.dev/news"
            payload = json.dumps({
                "q": f"{symbol} stock news",
                "num": num_articles
            })
            headers = {
                'X-API-KEY': os.getenv("SERPER_API_KEY"),
                'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)
            news_data = response.json().get('news', [])

            # 分析新闻情绪
            sentiment_prompt = PromptTemplate(
                input_variables=["articles"],
                template="""
                你是一位专业的金融新闻分析师，请分析以下股票新闻的情绪：
                {articles}

                请为每篇文章提供：
                1. 情绪评分（-10 到 10，负数为负面，正数为正面）
                2. 关键影响摘要
                3. 对股票价格的潜在影响（负面/中性/正面）

                最后，提供整体情绪分析总结。
                """
            )

            sentiment_chain = LLMChain(
                llm=deepseek_llm,
                prompt=sentiment_prompt,
                output_key="sentiment_analysis"
            )

            articles_text = "\n\n".join([
                f"标题: {item['title']}\n内容: {item['snippet']}\n来源: {item['source']}"
                for item in news_data
            ])

            sentiment_result = sentiment_chain.run(articles=articles_text)

            return {
                'news_articles': news_data,
                'sentiment_analysis': sentiment_result,
                'symbol': symbol,
                'num_articles': num_articles
            }
        except Exception as e:
            return {"error": str(e)}


class RiskAssessmentTool(BaseTool):
    name = "risk_assessor"
    description = "评估交易策略的风险并提供缓解建议"

    def _run(self, strategy: str, market_data: dict, capital: float) -> dict:
        """评估交易策略的风险"""
        try:
            # 创建风险评估链
            risk_prompt = PromptTemplate(
                input_variables=["strategy", "market_data", "capital"],
                template="""
                作为风险管理专家，请评估以下交易策略的风险：

                策略描述：
                {strategy}

                市场数据摘要：
                {market_data}

                初始资本：${capital:,.2f}

                请分析以下风险维度：
                1. 市场风险（系统性风险）
                2. 流动性风险
                3. 波动性风险
                4. 黑天鹅事件风险
                5. 特定股票风险

                为每种风险提供：
                - 概率评估（低/中/高）
                - 潜在影响程度（1-10分）
                - 具体缓解措施

                最后，提供整体风险评分（1-10分）和投资建议。
                """
            )

            risk_chain = LLMChain(
                llm=deepseek_llm,
                prompt=risk_prompt,
                output_key="risk_assessment"
            )

            # 简化市场数据用于提示
            simplified_data = {
                'symbol': market_data.get('symbol', ''),
                'current_price': market_data.get('current_price', 0),
                'volatility': market_data.get('volatility', 0),
                'rsi': market_data.get('rsi', 0),
                'market_cap': market_data.get('market_cap', 0),
                'pe_ratio': market_data.get('pe_ratio', 0)
            }

            return risk_chain.run({
                "strategy": strategy,
                "market_data": json.dumps(simplified_data, indent=2),
                "capital": capital
            })
        except Exception as e:
            return {"error": str(e)}


# 2. 创建工具实例
financial_data_tool = FinancialDataTool()
news_analysis_tool = NewsAnalysisTool()
risk_assessment_tool = RiskAssessmentTool()
scrape_tool = ScrapeWebsiteTool()

# 搜索工具
search = SerpAPIWrapper()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="用于搜索当前事件或特定信息"
)


# 3. 定义分析代理
def create_analysis_agent(name, role, goal, tools, prompt_template):
    """创建分析代理"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input", "agent_scratchpad"]
    )

    agent = create_react_agent(
        llm=deepseek_llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )


# 数据分析代理
data_analyst_agent = create_analysis_agent(
    name="DataAnalyst",
    role="金融数据分析师",
    goal="监控和分析市场数据，识别趋势并预测市场走势",
    tools=[financial_data_tool, news_analysis_tool, search_tool],
    prompt_template="""
    你是一位专业的金融数据分析师，负责持续监控和分析股票市场数据。
    使用可用工具获取最新的财务数据和新闻，然后进行分析。

    你的任务：
    1. 获取 {stock} 的历史价格和技术指标
    2. 分析近期市场趋势和模式
    3. 获取并分析相关新闻情绪
    4. 预测未来24-72小时的价格走势
    5. 识别关键风险和机会

    用户参数：
    - 初始资本: ${capital}
    - 风险承受能力: {risk_tolerance}
    - 交易策略偏好: {trading_preference}

    请提供包含以下内容的详细报告：
    - 技术分析摘要
    - 基本面分析摘要
    - 新闻情绪分析
    - 短期价格预测
    - 关键风险提示

    开始分析：
    {input}
    {agent_scratchpad}
    """
)

# 交易策略代理
strategy_agent = create_analysis_agent(
    name="StrategyDeveloper",
    role="交易策略开发专家",
    goal="基于分析洞察开发多种交易策略",
    tools=[financial_data_tool, scrape_tool, search_tool],
    prompt_template="""
    你是一位量化交易策略开发专家，基于以下输入开发交易策略：

    输入数据：
    - 目标股票: {stock}
    - 风险承受能力: {risk_tolerance}
    - 交易偏好: {trading_preference}
    - 初始资本: ${capital}
    - 市场分析: {market_analysis}

    你的任务：
    1. 开发多种交易策略（日内、波段、趋势跟踪等）
    2. 每种策略包含入场/出场条件
    3. 指定仓位管理规则
    4. 设置风险管理参数
    5. 估算预期回报

    策略要求：
    - 符合用户的风险承受能力
    - 适应用户的交易偏好
    - 考虑当前市场条件

    请提供包含以下内容的详细报告：
    - 策略1：详细描述和规则
    - 策略2：详细描述和规则
    - 策略比较和推荐

    开始开发策略：
    {input}
    {agent_scratchpad}
    """
)

# 交易执行代理
execution_agent = create_analysis_agent(
    name="ExecutionAdvisor",
    role="交易执行顾问",
    goal="制定最佳交易执行计划",
    tools=[financial_data_tool, search_tool],
    prompt_template="""
    作为交易执行专家，请为以下策略制定详细的交易执行计划：

    输入数据：
    - 目标股票: {stock}
    - 初始资本: ${capital}
    - 选定策略: {selected_strategy}
    - 市场分析: {market_analysis}

    你的任务：
    1. 确定最佳入场时机
    2. 建议订单类型（市价单、限价单等）
    3. 制定仓位分配计划
    4. 设置止损和止盈水平
    5. 规划退出策略

    执行计划应考虑：
    - 当前市场流动性
    - 波动性条件
    - 重大经济事件日历
    - 交易成本影响

    请提供包含以下内容的详细报告：
    - 具体执行步骤
    - 价格触发点
    - 应急计划
    - 预期结果

    开始制定执行计划：
    {input}
    {agent_scratchpad}
    """
)

# 风险管理代理
risk_agent = create_analysis_agent(
    name="RiskManager",
    role="风险管理专家",
    goal="评估交易风险并提供缓解措施",
    tools=[risk_assessment_tool, news_analysis_tool, search_tool],
    prompt_template="""
    作为风险管理专家，请评估以下交易计划的风险：

    输入数据：
    - 目标股票: {stock}
    - 交易策略: {trading_strategy}
    - 执行计划: {execution_plan}
    - 市场分析: {market_analysis}
    - 初始资本: ${capital}
    - 风险承受能力: {risk_tolerance}

    你的任务：
    1. 识别潜在风险（市场、流动性、操作等）
    2. 评估风险概率和影响
    3. 提供风险缓解措施
    4. 建议压力测试情景
    5. 提供整体风险评估

    请提供包含以下内容的详细报告：
    - 风险矩阵（概率 vs 影响）
    - 关键风险分析
    - 具体缓解建议
    - 整体风险评分（1-10分）

    开始风险评估：
    {input}
    {agent_scratchpad}
    """
)


# 4. 创建分析工作流
class FinancialAnalysisWorkflow:
    """金融交易分析工作流"""

    def __init__(self, inputs):
        self.inputs = inputs
        self.context = {}

    def run_data_analysis(self):
        """执行市场数据分析"""
        print("\n=== 开始市场数据分析 ===")
        result = data_analyst_agent.invoke({
            "input": f"分析 {self.inputs['stock']} 的市场数据",
            "stock": self.inputs["stock"],
            "capital": float(self.inputs["capital"]),
            "risk_tolerance": self.inputs["risk_tolerance"],
            "trading_preference": self.inputs["trading_preference"]
        })
        self.context["market_analysis"] = result["output"]
        print("=== 市场数据分析完成 ===")
        return result["output"]

    def develop_trading_strategy(self):
        """开发交易策略"""
        print("\n=== 开始交易策略开发 ===")
        result = strategy_agent.invoke({
            "input": f"为 {self.inputs['stock']} 开发交易策略",
            "stock": self.inputs["stock"],
            "capital": float(self.inputs["capital"]),
            "risk_tolerance": self.inputs["risk_tolerance"],
            "trading_preference": self.inputs["trading_preference"],
            "market_analysis": self.context["market_analysis"]
        })
        self.context["trading_strategies"] = result["output"]
        print("=== 交易策略开发完成 ===")
        return result["output"]

    def plan_execution(self):
        """制定执行计划"""
        print("\n=== 开始制定执行计划 ===")
        result = execution_agent.invoke({
            "input": f"为 {self.inputs['stock']} 制定交易执行计划",
            "stock": self.inputs["stock"],
            "capital": float(self.inputs["capital"]),
            "selected_strategy": self.context["trading_strategies"],
            "market_analysis": self.context["market_analysis"]
        })
        self.context["execution_plan"] = result["output"]
        print("=== 执行计划制定完成 ===")
        return result["output"]

    def assess_risk(self):
        """评估风险"""
        print("\n=== 开始风险评估 ===")
        result = risk_agent.invoke({
            "input": f"评估 {self.inputs['stock']} 交易风险",
            "stock": self.inputs["stock"],
            "trading_strategy": self.context["trading_strategies"],
            "execution_plan": self.context["execution_plan"],
            "market_analysis": self.context["market_analysis"],
            "capital": float(self.inputs["capital"]),
            "risk_tolerance": self.inputs["risk_tolerance"]
        })
        self.context["risk_assessment"] = result["output"]
        print("=== 风险评估完成 ===")
        return result["output"]

    def generate_final_report(self):
        """生成最终报告"""
        print("\n=== 生成最终分析报告 ===")

        report = f"""
        ===== 金融交易分析最终报告 =====
        生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        股票代码: {self.inputs['stock']}
        初始资本: ${float(self.inputs['capital']):,.2f}
        风险偏好: {self.inputs['risk_tolerance']}
        交易策略: {self.inputs['trading_preference']}

        === 市场分析摘要 ===
        {self.context.get('market_analysis', '')[:1000]}...

        === 交易策略推荐 ===
        {self.context.get('trading_strategies', '')[:1000]}...

        === 执行计划 ===
        {self.context.get('execution_plan', '')[:1000]}...

        === 风险评估 ===
        {self.context.get('risk_assessment', '')[:1000]}...

        === 综合建议 ===
        1. 推荐策略: [策略名称]
        2. 初始仓位: [建议仓位比例]%
        3. 关键入场点: [价格水平]
        4. 止损水平: [价格水平]
        5. 目标价位: [价格水平]
        6. 监控指标: [关键指标]

        注: 完整报告包含详细分析和支持数据
        """

        # 保存报告到文件
        filename = f"{self.inputs['stock']}_交易分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"报告已保存到: {filename}")
        return report


# 5. 可视化工具
def plot_stock_data(symbol, period="6mo"):
    """可视化股票数据"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)

        plt.figure(figsize=(14, 7))

        # 价格图表
        plt.subplot(2, 1, 1)
        plt.plot(hist.index, hist['Close'], label='收盘价', color='blue')
        plt.title(f'{symbol} 价格走势 ({period})')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)

        # 成交量图表
        plt.subplot(2, 1, 2)
        plt.bar(hist.index, hist['Volume'], color='green', alpha=0.7)
        plt.title('成交量')
        plt.ylabel('成交量')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{symbol}_price_volume.png")
        print(f"价格和成交量图表已保存: {symbol}_price_volume.png")

        # 技术指标图表（如果有足够数据）
        if len(hist) > 50:
            plt.figure(figsize=(14, 10))

            # 价格和移动平均线
            plt.subplot(3, 1, 1)
            plt.plot(hist.index, hist['Close'], label='收盘价', color='blue')
            plt.plot(hist.index, hist['SMA_50'], label='50日移动平均', color='orange')
            plt.plot(hist.index, hist['SMA_200'], label='200日移动平均', color='red')
            plt.title(f'{symbol} 价格与移动平均线')
            plt.ylabel('价格')
            plt.legend()
            plt.grid(True)

            # RSI
            plt.subplot(3, 1, 2)
            plt.plot(hist.index, hist['RSI'], label='RSI', color='purple')
            plt.axhline(70, color='red', linestyle='--')
            plt.axhline(30, color='green', linestyle='--')
            plt.title('相对强弱指数 (RSI)')
            plt.ylabel('RSI')
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(True)

            # MACD
            exp12 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp26 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()

            plt.subplot(3, 1, 3)
            plt.plot(hist.index, macd, label='MACD', color='blue')
            plt.plot(hist.index, signal, label='信号线', color='red')
            plt.bar(hist.index, macd - signal, color=np.where(macd - signal > 0, 'green', 'red'), alpha=0.5)
            plt.title('MACD指标')
            plt.ylabel('MACD')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"{symbol}_technical_indicators.png")
            print(f"技术指标图表已保存: {symbol}_technical_indicators.png")

        return True
    except Exception as e:
        print(f"图表生成错误: {str(e)}")
        return False


# 6. 主函数
def main():
    """主工作流"""
    # 用户输入
    analysis_inputs = {
        'stock': 'AAPL',
        'capital': '100000',
        'risk_tolerance': '中等',
        'trading_preference': '日内交易'
    }

    print("🚀 启动金融交易分析系统")
    print(f"分析目标: {analysis_inputs['stock']}")
    print(f"初始资本: ${float(analysis_inputs['capital']):,.2f}")
    print(f"风险偏好: {analysis_inputs['risk_tolerance']}")
    print(f"交易策略: {analysis_inputs['trading_preference']}")

    # 创建分析工作流
    workflow = FinancialAnalysisWorkflow(analysis_inputs)

    # 执行分析步骤
    workflow.run_data_analysis()
    workflow.develop_trading_strategy()
    workflow.plan_execution()
    workflow.assess_risk()

    # 生成报告
    report = workflow.generate_final_report()

    # 生成图表
    plot_stock_data(analysis_inputs['stock'])

    # 打印最终报告摘要
    print("\n" + "=" * 60)
    print("✅ 分析完成! 最终报告摘要:")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()