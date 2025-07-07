from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import random
from dotenv import load_dotenv
import os
from Docu2data import Split_Character, Split_NLTK, Recursive_split
from langchain_community.document_loaders import PyPDFLoader
from Embedding_model import create_embeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7  # 控制语言模型输出内容 随机性 的参数，官方推荐0.7
)


"""
使用LLM评估分块质量
返回: (平均分, 详细评估报告)
"""
def evaluate_chunks_with_llm(chunks):
    parser = StrOutputParser()

    # 2. 创建评估提示模板
    prompt = ChatPromptTemplate.from_template(
        "作为文档处理专家，请评估以下文本块的质量（1-5分）:\n"
        "评分标准:\n"
        "- 5分: 完美 - 语义完整，边界合理\n"
        "- 4分: 良好 - 基本完整，少量不连贯\n"
        "- 3分: 一般 - 部分内容断裂\n"
        "- 2分: 较差 - 明显碎片化\n"
        "- 1分: 很差 - 完全碎片化\n\n"
        "- 0分: 出错 - 没有检测到文本块\n\n"
        "文本块内容:\n"
        "-------------\n"
        "{chunk_content}\n"
        "-------------\n\n"
        "请先简要分析，然后直接给出最终评分（仅数字）"
    )

    # 3. 构建评估链
    evaluator = prompt | llm | parser

    # 4. 随机抽样评估（最多5个块）
    sample_size = min(5, len(chunks))
    sample_indices = random.sample(range(len(chunks)), sample_size)

    # 5. 执行评估,只评分  可以添加report = []查看详细报告，增加可解释性
    scores = []

    for idx in sample_indices:
        chunk = chunks[idx]
        content = chunk.page_content

        # 截断长内容（保留头尾），chunk可能过长，浪费资源
        if len(content) > 1500:
            content = content[:700] + "\n\n...[内容截断]...\n\n" + content[-700:]

        # 获取评估结果
        response = evaluator.invoke({"chunk_content": content})

        # 提取评分
        try:
            # 查找最后出现的数字
            score_str = response.strip().split()[-1]
            score = int(score_str)
            if 1 <= score <= 5:
                scores.append(score)
            else:
                score = 100  # 默认值
        except:
            score = 100  # 解析失败时使用默认值

    # 6. 计算平均分
    avg_score = sum(scores) / len(scores) if scores else 0

    return avg_score


'''加载PDF
'''
loader = PyPDFLoader("E:/Pycharm/LangChain/LangChain/RAG_Data/XXX.pdf")
data = loader.load()


'''选择评分最高的chunk
'''
def choose_chunk():
    chunks_Character = Split_Character(data)
    chunks_NLTK = Split_NLTK(data)
    chunks_Recursive = Recursive_split(data)

    avg_score1 = evaluate_chunks_with_llm(chunks_Character)
    avg_score2 = evaluate_chunks_with_llm(chunks_NLTK)
    avg_score3 = evaluate_chunks_with_llm(chunks_Recursive)

    # 使用元组和max函数找到最大分数对应的索引
    scores = (avg_score1, avg_score2, avg_score3)
    chunks_list = [chunks_Character, chunks_NLTK, chunks_Recursive]
    max_index = scores.index(max(scores))
    print(chunks_list[max_index])

    return chunks_list[max_index]

FAISS_INDEX_PATH = "E:/Pycharm/LangChain/FAISS_Stock_Data"
# 定义索引名，英文字符
INDEX_NAME = "jianli"


'''chunk向量化并存储FAISS
'''
def data_embedding():
    embedding = create_embeddings()
    chunk = choose_chunk()

    db = FAISS.from_documents(chunk, embedding)

    # 保存更新后的索引
    db.save_local(FAISS_INDEX_PATH, index_name=INDEX_NAME)
    print(f"索引已保存到: {FAISS_INDEX_PATH}")


# if __name__ == "__main__":
#     data_embedding()
