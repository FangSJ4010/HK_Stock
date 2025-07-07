from transformers import pipeline
import torch
import numpy as np
from langchain_core.embeddings import Embeddings
import warnings

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
#
# # 加载 .env 文件
# load_dotenv()
#
# api_key = os.getenv("DEEPSEEK_API_KEY")
#
#
# # 封装 OpenAI 或兼容 API 成为一个 LangChain 可用的语言模型接口（LLM）
# llm = ChatOpenAI(
#     api_key=api_key,
#     base_url="https://api.deepseek.com",
#     model="deepseek-chat",
#     temperature=0.7  # 控制语言模型输出内容 随机性 的参数，官方推荐0.7
# )



