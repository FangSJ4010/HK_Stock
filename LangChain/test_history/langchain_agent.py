from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chains import LLMMathChain
from langchain_community.utilities import AlphaVantageAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from datetime import datetime

# 1. 创建自定义工具 - 市场情绪分析
def market_sentiment_analyzer(query: str) -> str:
    """使用NLP分析市场情绪，返回积极/消极比例"""
    # 实际应用中可接入专业API如Bloomberg/路透
    return f"市场情绪分析 ({datetime.today().strftime('%Y-%m-%d')}):\n- 积极情绪: 62%\n- 消极情绪: 38%\n主要担忧: 利率政策"

# 2. 初始化API工具
alpha_vantage = AlphaVantageAPIWrapper(alphavantage_api_key="YOUR_API_KEY")
llm_math = LLMMathChain.from_llm(llm=ChatOpenAI(temperature=0))

# 3. 定义工具集
tools = [
    Tool(
        name="股票价格查询",
        func=alpha_vantage.run,
        description="获取实时股票价格，输入格式：'股票代码' 或 '公司名称 股票'"
    ),
    Tool(
        name="财务指标计算",
        func=llm_math.run,
        description="计算财务指标如PE比率、收益率等，输入数学表达式"
    ),
    Tool(
        name="市场情绪分析",
        func=market_sentiment_analyzer,
        description="分析当前市场整体情绪和风险因素"
    ),
    Tool(
        name="行业对比",
        func=lambda sector: f"{sector}行业对比数据：\n- 平均PE: 24.3\n- 增长率: 8.2%\n- 龙头股: AAPL, MSFT",
        description="获取不同行业的对比数据，输入行业名称"
    )
]

# 4. 创建带记忆的Agent
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,  # 保留最近5轮对话
    return_messages=True
)

financial_analyst = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model="gpt-4-turbo", temperature=0.3),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=5,
    agent_kwargs={
        "prefix": """你是一位资深金融分析师，需要：
        1. 精准回答股票和金融市场相关问题
        2. 结合实时数据和历史对话提供建议
        3. 当数据不足时主动使用工具获取信息
        4. 所有建议必须基于可靠数据"""
    }
)

# 5. 运行示例
queries = [
    "苹果公司当前股价是多少？",
    "计算如果我现在投资1万美元，按当前增长率3年后价值多少？",
    "对比科技和能源行业的投资价值",
    "结合当前市场情绪，给出AAPL的短期操作建议"
]

for query in queries:
    print(f"\n\033[1;34m用户提问: {query}\033[0m")
    response = financial_analyst.invoke({"input": query})
    print(f"\033[1;32m分析师回复: {response['output']}\033[0m")