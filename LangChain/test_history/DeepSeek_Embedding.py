import os
import requests
from typing import List
from langchain.embeddings.base import Embeddings

class DeepSeekEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "deepseek-embedding-v2"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.deepseek.com/v1/embeddings"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        json_data = {
            "model": self.model,
            "input": text
        }
        response = requests.post(self.api_url, headers=headers, json=json_data)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
