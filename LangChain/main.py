from HK_Data_Tool import hk_code_data, hk_tool
import os
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI  # 更新为新的导入方式
from dotenv import load_dotenv
from Data_Analyse_Tool import Analyse_Tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re

# 加载环境变量
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 初始化LLM - 使用新的导入方式避免警告
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7
)

# ===== 1. 创建数据获取Agent =====
# 使用新的代理创建方式避免警告
get_hk_data_Agent = initialize_agent(
    llm=llm,
    tools=[hk_tool],
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True  # 添加错误处理
)

# ===== 2. 创建数据分析Agent =====
system_prompt = """（保持原样）"""
Analyse_stock_Agent = initialize_agent(
    prompt=ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ]),
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=Analyse_Tools,
    verbose=True,
    handle_parsing_errors=True  # 添加错误处理
)


# ===== 3. 创建股票名称提取函数 =====
def extract_stock_names(input_dict: dict) -> list:
    """从用户查询中提取股票名称"""
    query = input_dict.get("query", "")
    if not query:
        return []

    # 简单匹配中文/英文/数字组合
    pattern = r'[\u4e00-\u9fa5a-zA-Z0-9]+'
    potential_names = re.findall(pattern, query)

    # 过滤掉常见非股票词汇（可根据需要扩展）
    common_words = ["股票", "分析", "走势", "对比", "数据", "获取", "查询", "一下", "帮我", "并", "的", "和"]
    return [name for name in potential_names if name not in common_words]


# ===== 4. 创建链式工作流 =====
# 步骤1: 提取股票名称
stock_extractor = RunnableLambda(extract_stock_names)

# 步骤2: 数据获取工作流
data_fetch_chain = (
        RunnablePassthrough.assign(stock_names=stock_extractor)
        | RunnableLambda(lambda x: {
    "input": f"获取{', '.join(x['stock_names'])}的最新数据" if x['stock_names'] else "没有需要获取的股票",
    "stock_names": x["stock_names"]
})
        | RunnableLambda(lambda x: get_hk_data_Agent.run(x["input"]) if x['stock_names'] else "没有执行数据获取")
        | RunnablePassthrough.assign(stock_names=lambda x: x["stock_names"])
)

# 步骤3: 数据分析工作流
analysis_chain = (
        RunnablePassthrough.assign(original_query=lambda x: x["query"])
        | RunnableLambda(lambda x: Analyse_stock_Agent.run(x["original_query"]))
)

# 完整工作流
full_workflow = (
        RunnablePassthrough.assign(
            fetched_data=lambda x: data_fetch_chain.invoke({"query": x["query"]})
        )
        | analysis_chain
        | StrOutputParser()
)

# ===== 5. 使用示例 =====
if __name__ == "__main__":
    # 示例查询 - 确保是字典格式
    query = {"query": "帮我查一下腾讯的详细信息，并分析近期走势和风险"}

    # 执行完整工作流
    try:
        print("\n" + "=" * 50)
        print("🚀 开始处理查询...")
        print("=" * 50)

        result = full_workflow.invoke(query)

        print("\n" + "=" * 50)
        print("💡 最终分析结果:")
        print("=" * 50)
        print(result)
    except Exception as e:
        print("\n" + "=" * 50)
        print("❌ 处理过程中出错:")
        print("=" * 50)
        print(str(e))
        print("\n请检查日志获取更多详细信息")