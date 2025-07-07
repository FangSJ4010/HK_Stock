import os
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_openai import ChatOpenAI
from crewai.process import Process
# 参考地址：https://blog.csdn.net/m0_59235245/article/details/143441635




# 获取 API 密钥（假设已存储在环境变量中）
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()


'''
数据分析Agent:持续监控选定股票（的市场数据，利用统计模型和机器学习算法，识别市场趋势并预测未来走势。
'''
data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time to identify trends and predict market movements.",
    backstory="Specializing in financial markets, this agent uses statistical modeling and machine learning to provide crucial insights.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

# 交易策略
trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and test various trading strategies based on insights from the Data Analyst Agent.",
    backstory="Equipped with a deep understanding of financial markets and quantitative analysis, this agent devises and refines trading strategies.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

# 交易顾问
execution_agent = Agent(
    role="Trade Advisor",
    goal="Suggest optimal trade execution strategies based on approved trading strategies.",
    backstory="This agent specializes in analyzing the timing, price, and logistical details of potential trades.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

# 风险管理
risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks associated with potential trading activities.",
    backstory="Armed with a deep understanding of risk assessment models and market dynamics, this agent scrutinizes the potential risks of proposed trades.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)





'''
创建数据分析任务
'''
data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market data for the selected stock ({stock_selection}). "
        "Use statistical modeling and machine learning to identify trends and predict market movements."
    ),
    expected_output=(
        "Insights and alerts about significant market opportunities or threats for {stock_selection}."
    ),
    agent=data_analyst_agent,
)

strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on the insights from the Data Analyst and user-defined risk tolerance ({risk_tolerance}). " 
        "Consider trading preferences ({trading_strategy_preference})."
    ),
    expected_output=(
        "A set of potential trading strategies for {stock_selection} that align with the user's risk tolerance."
    ),
    agent=trading_strategy_agent,
)

execution_planning_task = Task(
    description=(
        "Analyze approved trading strategies to determine the best execution methods for {stock_selection}, "  
        "considering current market conditions and optimal pricing."       ),
    expected_output=(
        "Detailed execution plans suggesting how and when to execute trades for {stock_selection}."
    ),
    agent=execution_agent,
)

risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed trading strategies and execution plans for {stock_selection}. "
        "Provide a detailed analysis of potential risks and suggest mitigation strategies."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential risks and mitigation recommendations for {stock_selection}."
    ),
    agent=risk_management_agent,
)

'''
管理者与组建团队
'''
financial_trading_crew = Crew(
    agents=[
        data_analyst_agent,
        trading_strategy_agent,
        execution_agent,
        risk_management_agent
    ],
    tasks=[
        data_analysis_task,
        strategy_development_task,
        execution_planning_task,
        risk_assessment_task
    ],
    manager_llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
    process=Process.hierarchical,
    verbose=True
)

'''
运行系统
'''
financial_trading_inputs = {
    'stock_selection': 'AAPL',
    'initial_capital': '100000',
    'risk_tolerance': 'Medium',
    'trading_strategy_preference': 'Day Trading',
    'news_impact_consideration': True
}


'''
启动系统
'''
result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)

'''
显示结果
'''
from IPython.display import Markdown
Markdown(result)



