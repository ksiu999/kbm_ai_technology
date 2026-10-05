import json
import os
import re
import time
import streamlit as st
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from config import (
    ROUTER_BASE_URL,
    ROUTER_MODEL,
    SCHOOL_BASE_URL,
    SCHOOL_EMBEDDING_MODEL,
)

load_dotenv()


# --- Косинусное сходство ---
def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# --- Конфигурация API ---
EMBED_URL = f"{SCHOOL_BASE_URL}/embeddings"
CHAT_URL = f"{ROUTER_BASE_URL}/chat/completions"


# --- Функция получения эмбеддинга (Школьный API) ---
def get_query_embedding(text, max_retries=3):
    """Генерирует эмбеддинг пользовательского запроса через HTTP-запрос."""
    api_key = os.getenv("SCHOOL_API_KEY")
    if not api_key:
        raise Exception("SCHOOL_API_KEY не найден в .env")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                EMBED_URL,
                json={"model": SCHOOL_EMBEDDING_MODEL, "input": text},
                headers=headers,
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
            if resp.status_code >= 500:
                time.sleep((attempt + 1) * 2)
                continue
            raise Exception(f"Embedding Error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep((attempt + 1) * 2)


# --- Проверка API ключей ---
school_key = os.getenv("SCHOOL_API_KEY")
router_key = os.getenv("ROUTER_API_KEY")

if not school_key:
    st.error("❌ SCHOOL_API_KEY не найден в .env")
else:
    st.sidebar.success("\u2705 School API Key loaded")

if not router_key:
    st.error("❌ ROUTER_API_KEY не найден в .env")
else:
    st.sidebar.success("\u2705 RouterAI Key loaded")

st.caption(
    f"🟢 Chat: {ROUTER_MODEL} via RouterAI | 🔵 Embeddings: {SCHOOL_EMBEDDING_MODEL} via School API"
)

# --- Загрузка базы знаний ---
DB_PATH = "db/chunks.jsonl"
st.title("\U0001f3ed Агент расчета оборудования КБМ")

if os.path.exists(DB_PATH):
    with open(DB_PATH, "r", encoding="utf-8") as f:
        st.session_state.chunks = [json.loads(line) for line in f if line.strip()]
    st.sidebar.success(f"\U0001f4da База знаний: {len(st.session_state.chunks)} чанков")
else:
    st.warning(
        f"База данных не найдена: {DB_PATH}. Запустите `python3 preprocess_kb.py`"
    )
    st.session_state.chunks = []

# Чат
if len(st.session_state.chunks) == 0:
    prompt = st.chat_input("Задайте вопрос технологу...", disabled=True)
else:
    prompt = st.chat_input("Задайте вопрос технологу...")

if prompt:
    # ===== ШАГ 1: Поиск похожих чанков =====
    with st.spinner("Ищем данные в базе..."):
        try:
            query_emb = get_query_embedding(prompt)
        except Exception as e:
            st.error(f"Ошибка векторизации: {e}")
            st.stop()

    ranked = []
    for chunk in st.session_state.chunks:
        sim = cosine_similarity(query_emb, chunk["embedding"])
        ranked.append((chunk, float(sim)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_chunks = ranked[:5]

    # ===== ШАГ 2: Подготовка контекста =====
    context_parts = []
    for i, (chunk, sim) in enumerate(top_chunks):
        context_parts.append(
            f"[Источник: {chunk['source']}]\n"
            f"Чанк #{i + 1} (схожесть: {sim:.4f}):\n{chunk['text']}"
        )
    context = "\n---\n".join(context_parts)

    system_prompt = (
        "Ты инженер-технолог КБМ. Отвечай кратко и по существу, опираясь ТОЛЬКО на предоставленный КОНТЕКСТ. "
        "Если информации нет, так и скажи. "
        "Ответ СТРОГО в формате JSON: "
        "{ 'summary': 'текстовый ответ', 'equipment_table': [{'name': 'название', 'qty': число, 'spec': 'характеристика'}] }."
    )

    # ===== ШАГ 3: Генерация ответа через RouterAI =====
    col_chat, col_result, col_sources = st.columns([1, 1.5, 1])

    with col_chat:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            msg_placeholder = st.empty()

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Контекст:\n{context}\n\nВопрос: {prompt}",
                },
            ]

            payload = {"model": ROUTER_MODEL, "messages": messages, "temperature": 0.0}

            headers = {
                "Authorization": f"Bearer {router_key}",
                "Content-Type": "application/json",
            }

            try:
                with st.spinner(f"Генерация ответа ({ROUTER_MODEL})..."):
                    r = requests.post(
                        CHAT_URL, json=payload, headers=headers, timeout=60
                    )

                if r.status_code != 200:
                    st.error(f"Ошибка RouterAI: HTTP {r.status_code}")
                    st.code(r.text[:500])
                    st.stop()

                raw_answer = r.json()["choices"][0]["message"]["content"]

                # Парсинг JSON
                summary_text = raw_answer
                equipment_data = []

                try:
                    json_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_answer)
                    if json_block:
                        parsed = json.loads(json_block.group(1))
                    else:
                        parsed = json.loads(raw_answer)

                    if isinstance(parsed, dict) and "equipment_table" in parsed:
                        summary_text = parsed.get("summary", "")
                        equipment_data = parsed.get("equipment_table", [])
                    elif isinstance(parsed, list):
                        equipment_data = parsed

                except json.JSONDecodeError:
                    st.warning("⚠️ Модель вернула не JSON, вывожу сырой текст.")

                msg_placeholder.markdown(summary_text if summary_text else raw_answer)

            except Exception as e:
                st.error(f"Ошибка соединения с RouterAI: {e}")

    # — Колонка 2: Таблица результата —
    with col_result:
        st.subheader("📊 Результат расчета")
        if equipment_data:
            table_rows = []
            for item in equipment_data:
                if isinstance(item, dict):
                    table_rows.append(
                        {
                            "Название": str(item.get("name", "")),
                            "Кол-во": str(item.get("qty", "")),
                            "Характеристика": str(item.get("spec", "")),
                        }
                    )
            if table_rows:
                df_equipment = pd.DataFrame(table_rows)
                st.dataframe(df_equipment, use_container_width=True, hide_index=True)
            else:
                st.info("Нет данных об оборудовании.")
        else:
            st.info("Структурированные данные не получены.")

    # — Колонка 3: Источники —
    with col_sources:
        st.subheader("📚 Источники RAG")
        with st.expander("Показать 5 найденных фрагментов"):
            for i, (chunk, sim) in enumerate(top_chunks):
                st.caption(f"#{i + 1} · схожесть: {sim:.4f}")
                truncated = chunk["text"][:400]
                if len(chunk["text"]) > 400:
                    truncated += "..."
                st.text_area("", value=truncated, height=200, disabled=True)
