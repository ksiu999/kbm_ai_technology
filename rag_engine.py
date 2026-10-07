import os
import re
import jsonlines
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from pymorphy3 import MorphAnalyzer
from razdel import tokenize
from nltk.corpus import stopwords

# Безопасная загрузка стоп-слов
try:
    STOP_WORDS = set(stopwords.words("russian"))
except LookupError:
    import nltk

    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("russian"))

MORPH = MorphAnalyzer()


def tokenize_russian(text: str) -> list[str]:
    """Токенизация, очистка и лемматизация русского текста."""
    tokens = []
    for word in tokenize(text):
        w = word.text.lower()
        # Оставляем только буквы и цифры
        w = re.sub(r"[^\w\s]", "", w)
        if w and w not in STOP_WORDS and len(w) > 1:
            # Лемматизация
            normal_form = MORPH.parse(w)[0].normal_form
            tokens.append(normal_form)
    return tokens


class HybridSearchEngine:
    def __init__(self, db_path: str, embed_url: str, api_key: str, alpha: float = 0.5):
        self.db_path = db_path
        self.embed_url = embed_url
        self.api_key = api_key
        self.alpha = alpha
        self.chunks = []
        self.bm25 = None
        self._load_index()

    def _load_index(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"База данных не найдена: {self.db_path}")

        with jsonlines.open(self.db_path, mode="r") as reader:
            self.chunks = list(reader)

        # Строим BM25 индекс на лемматизированных текстах
        tokenized_corpus = [tokenize_russian(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ Индекс загружен: {len(self.chunks)} чанков.")

    def _get_query_embedding(self, query: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": "text-embedding-3-small", "input": query}
        response = requests.post(self.embed_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def _normalize(self, scores: list[float]) -> list[float]:
        """Min-Max нормализация с защитой от деления на ноль."""
        mx, mn = max(scores), min(scores)
        if mx == mn:
            return [0.5] * len(scores)
        return [(s - mn) / (mx - mn) for s in scores]

    def search(self, query: str, top_k: int = 5) -> dict:
        keywords = tokenize_russian(query)
        query_embed = self._get_query_embedding(query)
        query_embed_np = np.array(query_embed)

        # 1. Embedding scores (Cosine Similarity)
        embed_scores = []
        for chunk in self.chunks:
            chunk_embed_np = np.array(chunk["embedding"])
            norm_q = np.linalg.norm(query_embed_np)
            norm_c = np.linalg.norm(chunk_embed_np)
            if norm_q == 0 or norm_c == 0:
                score = 0.0
            else:
                score = float(
                    np.dot(query_embed_np, chunk_embed_np) / (norm_q * norm_c)
                )
            embed_scores.append(score)

        # 2. BM25 scores
        bm25_scores = self.bm25.get_scores(keywords)

        # 3. Нормализация и фьюжн
        norm_embed = self._normalize(embed_scores)
        norm_bm25 = self._normalize(bm25_scores)

        final_scores = [
            (self.alpha * e) + ((1 - self.alpha) * b)
            for e, b in zip(norm_embed, norm_bm25)
        ]

        # 4. Сбор результатов
        results_with_scores = []
        for i, chunk in enumerate(self.chunks):
            results_with_scores.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "final_score": final_scores[i],
                    "embed_score": embed_scores[i],
                    "bm25_score": bm25_scores[i],
                }
            )

        # Сортировка по финальному скору
        results_with_scores.sort(key=lambda x: x["final_score"], reverse=True)
        top_final = results_with_scores[:top_k]

        # Для дебага: топ-3 чисто по embedding и чисто по bm25
        top_by_embed = sorted(
            results_with_scores, key=lambda x: x["embed_score"], reverse=True
        )[:3]
        top_by_bm25 = sorted(
            results_with_scores, key=lambda x: x["bm25_score"], reverse=True
        )[:3]

        return {
            "keywords": keywords,
            "top_embedding": top_by_embed,
            "top_bm25": top_by_bm25,
            "final_results": top_final,
        }
