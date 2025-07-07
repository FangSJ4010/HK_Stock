from transformers import pipeline
import torch
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import warnings
import os
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
# 过滤弃用警告
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")


# 自定义嵌入类
class CustomHuggingFaceEmbeddings(Embeddings):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def embed_documents(self, texts):
        results = self.pipeline(texts)
        embeddings = [np.array(result[0][0]).tolist() for result in results]
        return embeddings

    def embed_query(self, text):
        result = self.pipeline(text)
        return np.array(result[0][0]).tolist()


MODEL_PATH = "E:/Pycharm/LangChain/embedding"

# 创建特征提取 pipeline
feature_extractor = pipeline(
    "feature-extraction",
    model=MODEL_PATH,
    device=0 if torch.cuda.is_available() else -1,
    local_files_only=True
)


def create_embeddings():
    return CustomHuggingFaceEmbeddings(feature_extractor)

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


# ------------ RAG 集成部分 ------------
def setup_rag_system():
    # 1. 加载嵌入模型
    embeddings = create_embeddings()

    # 2. 加载预构建的FAISS向量库
    FAISS_INDEX_PATH = "E:/Pycharm/LangChain/FAISS_Stock_Data"

    # 修复加载问题，添加安全参数
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # 添加安全确认参数
    )

    print("======", vector_store)

    # 4. 创建RAG链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # 返回3个相关文档
        ),
        return_source_documents=True
    )

    return qa_chain


# 使用示例
if __name__ == "__main__":
    # 初始化RAG系统
    rag_system = setup_rag_system()

    # 用户提问
    while True:
        query = input("\n用户提问 (输入q退出): ")
        if query.lower() == 'q':
            break

        # 执行RAG查询
        result = rag_system.invoke({"query": query})

        print("\n回答:", result["result"])
        print("\n来源文档:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"{i + 1}. {doc.page_content[:100]}...")