from HK_Data_Tool import hk_code_data, hk_tool
import os
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
from Data_Analyse_Tool import Analyse_Tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 加载 .env 文件
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")


# 封装 OpenAI 或兼容 API 成为一个 LangChain 可用的语言模型接口（LLM）
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7  # 控制语言模型输出内容 随机性 的参数，官方推荐0.7
)

'''股票基本信息获取'''
get_hk_data_Agent = initialize_agent(
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=[hk_tool],
    verbose=True  # 显示 Agent 推理过程
)


'''根据股票信息进行分析'''
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
Analyse_stock_Agent = initialize_agent(
    # 创建提示模板
    prompt=ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ]),
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=Analyse_Tools,
    verbose=True  # 显示 Agent 推理过程
)


# response = get_hk_data_Agent.run("帮我查一下粤港湾的详细信息")
response = Analyse_stock_Agent.run("帮我查一下腾讯的详细信息，并分析一下形势")
print(response)