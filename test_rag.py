"""Быстрая проверка RAG-пайплайна: load chunks + embedding + top-k search."""
import json
import os
import re
import numpy as np
import requests

API_KEY = os.getenv("SCHOOL_API_KEY") or "uii_ai_VnEVVZlPavwIhTry7Ihc9nhPdjDyHVuMcOvYETg4Ktw"
EMBED_URL = "https://ai-api.neural-university.ru/v1/embeddings"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def clean(text):
    """Агрессивная чистка под API (такая же как в preprocess_kb.py)."""
    t = re.sub(r';+', ' ', text)
    t = re.sub(r'\s+', ' ', t).strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.-,:()!'\" /")
    t = "".join(c for c in t if c in allowed)
    return t.strip()

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_emb(text):
    payload = {"model": "text-embedding-3-small", "input": text}
    r = requests.post(EMBED_URL, json=payload, headers=HEADERS)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

# Загрузка базы
DB_PATH = "db/chunks.jsonl"
with open(DB_PATH, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]
print(f"📚 Загружено чанков: {len(chunks)}\n")

def search(query, top_k=5):
    q_emb = get_emb(query)
    scored = [(i, cosine_sim(q_emb, c["embedding"])) for i, c in enumerate(chunks)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

questions = [
    ("Какое оборудование используется для токарной обработки? Укажи название и количество.", 1),
    ("Какой инструмент указан для токарного станка DCL-30 при установке заготовки? Выпиши наименование патрона и его обозначение.", 2),
    ("Какие режимы резания указаны для операции подрезки торца? Укажи подачу число оборотов и время.", 3),
    ("Какой фрезерный центр применяется на уч 035? Перечисли модель и количество.", 4),
    ("Какая СОЖ используется при обработке? Напиши наименование.", 5),
    ("Какие инструменты и режущие элементы указаны для операции точения наружного контура? Выпиши марку резца.", 6),
    ("Какие средства измерений применяются для контроля размеров согласно эскизу? Перечисли все из одной строки инструментов.", 7),
    ("Какой ленточнопильный станок используется для отрезки заготовки? Укажи полное обозначение модели.", 8),
    ("Какие сверла применяются для операций сверления отверстий 10мм и 13мм? Укажи марки и характеристики каждого.", 9),
    ("Какая электроэрозионная установка используется для обработки тонкостенных стенок? Приведи модель и тип контроля.", 10),
]

for q, idx in questions:
    print(f"\n{'='*60}")
    print(f"❓ Вопрос #{idx}: {q}")
    print(f"{'='*60}")
    hits = search(clean(q), top_k=3)
    for rank, (chunk_idx, sim) in enumerate(hits, 1):
        text_preview = chunks[chunk_idx]["text"][:120].replace("\n", " ")
        print(f"  #{rank} | сходство={sim:.4f} | [{chunk_idx}] {text_preview}...")

print(f"\n✅ Тест завершён!")
