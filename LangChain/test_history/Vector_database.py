import dotenv
from langchain_community.vectorstores import FAISS
import os
from transformers import pipeline
import torch
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document


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


# 设置模型路径
MODEL_PATH = "E:/Pycharm/LangChain/embedding"

# 创建特征提取 pipeline
feature_extractor = pipeline(
    "feature-extraction",  # 修正拼写错误
    model=MODEL_PATH,
    device=0 if torch.cuda.is_available() else -1,
    local_files_only=True
)

# 创建自定义嵌入对象
embedding = CustomHuggingFaceEmbeddings(feature_extractor)

FAISS_INDEX_PATH = "E:/Pycharm/LangChain/FAISS_Data"  # 向量索引保存目录
INDEX_NAME = "猫咪测试"  # 索引名称

'''添加或更新向量库'''


def add_or_update_index(texts, index_name=INDEX_NAME):
    # 检查索引是否存在
    if os.path.exists(os.path.join(FAISS_INDEX_PATH, f"{index_name}.faiss")):
        # 加载已有索引
        db = FAISS.load_local(FAISS_INDEX_PATH, embedding, index_name=index_name)
        print(f"已加载现有索引 '{index_name}'，包含 {db.index.ntotal} 个文档")

        # 添加新文档
        new_docs = [Document(page_content=text) for text in texts]
        db.add_documents(new_docs)
        print(f"已添加 {len(texts)} 个新文档")
    else:
        # 创建新索引
        db = FAISS.from_texts(texts, embedding)
        print(f"创建新索引 '{index_name}'，包含 {len(texts)} 个文档")

    # 保存更新后的索引
    db.save_local(FAISS_INDEX_PATH, index_name=index_name)
    print(f"索引已保存到: {FAISS_INDEX_PATH}")

    return db


# 要添加的文本
texts = [
    "笨笨是一只很喜欢睡觉的猫咪",
    "我喜欢在夜晚听音乐，这让我感到放松。",
    "Cats napping on windowsills look very cute.",
    "Learning new skills is a goal everyone should pursue.",
    "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
    "Last night I had a strange dream about flying in space.",
    "My phone suddenly turned off, which made me anxious.",
    "Reading is something I do every day and it makes me feel fulfilled.",
    "They planned a weekend picnic together, hoping for good weather.",
    "My dog loves chasing balls and looks very happy.",
]

# 添加或更新索引
db = add_or_update_index(texts, INDEX_NAME)


#
# # 测试相似度搜索
# query = "一只懒惰的猫"
# results = db.similarity_search(query, k=3)  # 添加k参数限制结果数量
# print("\n相似度搜索结果:")
# for i, doc in enumerate(results):
#     print(f"{i + 1}. {doc.page_content}")
#
# # 添加更多文本的示例
# new_texts = [
#     "笨笨最近学会了开门，真是个聪明的小家伙",
#     "周末打算带狗狗去公园玩飞盘"
# ]
#
# print("\n添加新文本...")
# db = add_or_update_index(new_texts, INDEX_NAME)
#
# # 再次搜索
# print("\n更新后的相似度搜索结果:")
# results = db.similarity_search(query, k=3)
# for i, doc in enumerate(results):
#     print(f"{i + 1}. {doc.page_content}")