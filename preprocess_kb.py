#!/usr/bin/env python3
"""
Предобработка kbm_base.csv -> kbm_clean.txt + генерация чанков и эмбеддингов.
Результат сохраняется в db/chunks.jsonl для мгновенной загрузки в app.py.
"""
import json
import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Пути
SRC_CSV = "docs/kbm_base.csv"
OUT_TXT = "docs/kbm_clean.txt"
OUT_DB  = "db/chunks.jsonl"
MAX_RETRIES = 3
EMBED_MODEL = "text-embedding-3-small"
API_URL = "https://ai-api.neural-university.ru/v1/embeddings"
api_key = os.getenv("SCHOOL_API_KEY") or "uii_ai_VnEVVZlPavwIhTry7Ihc9nhPdjDyHVuMcOvYETg4Ktw"


def clean_line(line: str) -> str | None:
    """Агрессивная очистка под embeddings API."""
    t = re.sub(r';+', ' ', line)
    t = re.sub(r'\s+', ' ', t).strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.-,:()!'\" /")
    t = "".join(c for c in t if c in allowed)
    return t.strip() if len(t.strip()) > 0 else None


def get_embedding(text: str) -> list[float]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": EMBED_MODEL, "input": text}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers)
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
            # Retry on server errors
            if resp.status_code >= 500:
                wait = (attempt + 1) * 3
                print(f"\n  ⏳ Server {resp.status_code}, retry {attempt+1}/{MAX_RETRIES} in {wait}s...")
                time.sleep(wait)
                continue
            raise Exception(f"{resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.ConnectionError as e:
            print(f"\n  ⏳ Connection error, retry {attempt+1}/{MAX_RETRIES}...")
            time.sleep((attempt+1)*2)
        except Exception as e:
            if any(s in str(e) for s in ["500", "502", "503", "504"]):
                wait = (attempt + 1) * 3
                print(f"\n  ⏳ Error {e}, retry {attempt+1}/{MAX_RETRIES} in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded for embedding")


def main():
    print(f"📖 Читаем {SRC_CSV}...")
    with open(SRC_CSV, "r", encoding="utf-8-sig") as f:
        raw_lines = f.readlines()

    cleaned = []
    for i, line in enumerate(raw_lines):
        cl = clean_line(line)
        if cl:
            cleaned.append(cl)

    print(f"✅ Очистка: {len(raw_lines)} -> {len(cleaned)} строк.")

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned))
    print(f"💾 Сохранён {OUT_TXT}")

    chunks = []
    for idx, txt in enumerate(cleaned):
        if len(txt) <= 500:
            chunks.append({"id": idx, "text": txt})
        else:
            for seg_idx in range(0, len(txt), 500):
                chunks.append({"id": f"{idx}_{seg_idx//500}", "text": txt[seg_idx:seg_idx+500]})

    print(f"🧩 Всего чанков: {len(chunks)}")

    os.makedirs(os.path.dirname(OUT_DB), exist_ok=True)
    
    # Читаем существующую БД если есть (для инкрементальной записи)
    saved_ids = set()
    existing_records = []
    if os.path.exists(OUT_DB):
        with open(OUT_DB, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    saved_ids.add(rec["id"])
                    existing_records.append(rec)
        print(f"♻️ Найдено {len(saved_ids)} ранее сохранённых чанков, пропуск...")

    with open(OUT_DB, "a", encoding="utf-8") as db:
        for i, ch in enumerate(chunks):
            if ch["id"] in saved_ids:
                print(f"[{i+1}/{len(chunks)}] ⏭️ Пропуск (уже есть)")
                continue
                
            print(f"[{i+1}/{len(chunks)}] Эмбеддинг... ", end="", flush=True)
            try:
                vec = get_embedding(ch["text"])
                rec = {"id": ch["id"], "text": ch["text"], "embedding": vec, "source": "kbm_clean.txt"}
                db.write(json.dumps(rec, ensure_ascii=False) + "\n")
                saved_ids.add(ch["id"])
                print("✅")
            except Exception as e:
                print(f"❌ Пропуск: {type(e).__name__}: {str(e)[:80]}")

    final_count = out_db_check()
    print(f"✅ Готово! В БД: {final_count} записей.")


def out_db_check():
    if os.path.exists(OUT_DB):
        with open(OUT_DB, "r", encoding="utf-8") as f:
            return sum(1 for l in f if l.strip())
    return 0


if __name__ == "__main__":
    main()
