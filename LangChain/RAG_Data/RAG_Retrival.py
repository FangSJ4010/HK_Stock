from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from datetime import datetime
import numpy as np
from Embedding_model import create_embeddings

# 初始化嵌入模型
Embedding = create_embeddings()


# 加载FAISS向量库
def load_faiss_vectorstore(index_name=None):
    """根据是否提供index_name加载FAISS向量库"""
    FAISS_path = "E:/Pycharm/LangChain/FAISS_Stock_Data"

    if index_name:
        # 加载指定索引
        return FAISS.load_local(
            folder_path=FAISS_path,
            embeddings=Embedding,
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
    else:
        # 加载全部索引
        return FAISS.load_local(
            folder_path=FAISS_path,
            embeddings=Embedding,
            allow_dangerous_deserialization=True
        )


# ====================== 检索方法实现 ======================

def mmr_search(vectorstore, query, k=5, lambda_mult=0.7):
    """
    最大边际相关性检索
    - vectorstore: FAISS向量库
    - query: 查询文本
    - k: 返回结果数量
    - lambda_mult: 多样性控制参数 (0.5=平衡, 1.0=纯相似度, 0.0=纯多样性)
    """
    return vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        lambda_mult=lambda_mult
    )


def threshold_search(vectorstore, query, k=10, score_threshold=0.75):
    """
    带分数阈值检索
    - vectorstore: FAISS向量库
    - query: 查询文本
    - k: 初始检索数量
    - score_threshold: 相似度分数阈值
    """
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    return [doc for doc, score in docs_with_scores if score >= score_threshold]


def metadata_filter_search(vectorstore, query, filter_dict, k=5):
    """
    元数据过滤检索
    - vectorstore: FAISS向量库
    - query: 查询文本
    - filter_dict: 元数据过滤条件, 如 {"sector": "Technology"}
    - k: 返回结果数量
    """
    return vectorstore.similarity_search(
        query,
        k=k,
        filter=filter_dict
    )


def self_query_search(vectorstore, query, metadata_field_info, k=5):
    """
    自查询检索
    - vectorstore: FAISS向量库
    - query: 查询文本
    - metadata_field_info: 元数据字段描述
    - k: 返回结果数量
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="股票市场信息",
        metadata_field_info=metadata_field_info,
        verbose=False
    )
    return retriever.get_relevant_documents(query)[:k]


def time_weighted_search(vectorstore, query, time_field="date", alpha=0.1, k=5):
    """
    时间加权检索
    - vectorstore: FAISS向量库
    - query: 查询文本
    - time_field: 元数据中的时间字段
    - alpha: 时间权重系数
    - k: 返回结果数量
    """
    # 获取基础相似度结果
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=10)

    # 计算时间加权分数
    base_date = datetime.now()
    weighted_docs = []

    for doc, score in docs_with_scores:
        if time_field in doc.metadata:
            try:
                # 尝试解析不同格式的时间
                doc_date = datetime.strptime(doc.metadata[time_field], "%Y-%m-%d")
            except:
                try:
                    doc_date = datetime.strptime(doc.metadata[time_field], "%Y/%m/%d")
                except:
                    # 如果无法解析，使用当前时间
                    doc_date = base_date

            # 计算时间差异（月份）
            time_diff = (doc_date - base_date).total_seconds() / (3600 * 24 * 30)

            # 计算加权分数（时间越新分数越高）
            weighted_score = score + alpha * abs(time_diff)
            weighted_docs.append((doc, weighted_score))

    # 按加权分数排序
    weighted_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in weighted_docs[:k]]


def hyde_search(vectorstore, query, k=5):
    """
    假设文档嵌入检索 (HyDE)
    - vectorstore: FAISS向量库
    - query: 查询文本
    - k: 返回结果数量
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # 生成假设答案
    prompt = ChatPromptTemplate.from_template(
        "基于以下问题，生成一个假设性的详细答案:\n问题: {query}\n假设答案:"
    )
    hyde_chain = prompt | llm | StrOutputParser()
    hypothetical_answer = hyde_chain.invoke({"query": query})

    # 使用假设答案检索
    return vectorstore.similarity_search(hypothetical_answer, k=k)


def rerank_search(vectorstore, query, k=3, base_k=10):
    """
    重新排序检索
    - vectorstore: FAISS向量库
    - query: 查询文本
    - k: 最终返回结果数量
    - base_k: 初始检索数量
    """
    # 初始化交叉编码器
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=k)

    # 创建基础检索器
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": base_k})

    # 创建压缩检索器
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return compression_retriever.get_relevant_documents(query)


# ====================== 使用示例 ======================

def main():
    # 示例1: 加载全部向量库
    full_vectorstore = load_faiss_vectorstore()

    # 示例2: 加载指定索引
    # sector_vectorstore = load_faiss_vectorstore(index_name="sector_index")

    # 定义元数据字段信息 (用于自查询)
    metadata_field_info = [
        {"name": "company", "description": "公司名称", "type": "string"},
        {"name": "sector", "description": "行业分类", "type": "string"},
        {"name": "date", "description": "报告日期", "type": "date"}
    ]

    # 示例查询
    query = "近期科技板块表现优异的股票"

    print("1. MMR检索结果:")
    results = mmr_search(full_vectorstore, query, k=3, lambda_mult=0.6)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res.page_content[:60]}...")

    print("\n2. 自查询检索结果:")
    results = self_query_search(full_vectorstore, query, metadata_field_info, k=3)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res.page_content[:60]}...")

    print("\n3. 时间加权检索结果:")
    results = time_weighted_search(full_vectorstore, query, alpha=0.15, k=3)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res.page_content[:60]}...")

    print("\n4. HyDE检索结果:")
    results = hyde_search(full_vectorstore, query, k=3)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res.page_content[:60]}...")

    print("\n5. 重新排序检索结果:")
    results = rerank_search(full_vectorstore, query, k=3)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res.page_content[:60]}...")


if __name__ == "__main__":
    main()