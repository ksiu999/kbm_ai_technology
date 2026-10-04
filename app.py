import json
import os
import re
import time
import streamlit as st
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()


# --- Косинусное сходство ---
def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# --- Embedding через прямые requests (надёжнее, чем OpenAI SDK для gateway) ---
EMBED_URL = "https://ai-api.neural-university.ru/v1/embeddings"


def get_query_embedding(text, max_retries=3):
    """Генерирует эмбеддинг пользовательского запроса через HTTP-запрос."""
    api_key = os.getenv("SCHOOL_API_KEY") or "uii_ai_VnEVVZlPavwIhTry7Ihc9nhPdjDyHVuMcOvYETg4Ktw"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(EMBED_URL, json={"model": "text-embedding-3-small", "input": text}, headers=headers)
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
            # Server errors — ретраим
            if resp.status_code >= 500:
                wait = (attempt + 1) * 3
                print(f"\n⏳ Server {resp.status_code}, retry {attempt+1}/{max_retries} в {wait}с...")
                time.sleep(wait)
                continue
            raise Exception(f"{resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.ConnectionError:
            print(f"\n⏳ Connection error, retry {attempt+1}/{max_retries}...")
            time.sleep((attempt + 1) * 2)
        except Exception as e:
            err_str = str(e)
            if any(s in err_str for s in ["500", "502", "503", "504"]):
                wait = (attempt + 1) * 3
                print(f"\n⏳ Error: {e}, retry {attempt+1}/{max_retries} в {wait}с...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded")


# --- Проверка API ключа ---
api_key = os.getenv("SCHOOL_API_KEY")
if not api_key:
    st.error("API ключ не найден в .env")
else:
    st.sidebar.success("\u2705 API ключ загружен успешно")

# --- Загрузка пред-вычисленной базы знаний (без вызова API при запуске) ---
DB_PATH = "db/chunks.jsonl"
st.title("\U0001f3ed Агент расчета оборудования КБМ")

if os.path.exists(DB_PATH):
    with open(DB_PATH, "r", encoding="utf-8") as f:
        st.session_state.chunks = [json.loads(line) for line in f if line.strip()]
    st.sidebar.success(f"\U0001f4da База знаний: {len(st.session_state.chunks)} чанков загружена")
else:
    st.warning(f"База данных не найдена по пути {DB_PATH}. Запустите `python3 preprocess_kb.py` сначала.")
    st.session_state.chunks = []

# Чат
if len(st.session_state.chunks) == 0:
    prompt = st.chat_input("Задайте вопрос технологу...", disabled=True)
else:
    prompt = st.chat_input("Задайте вопрос технологу...")

if prompt:
    # ===== ШАГ 2: Эмбеддинг промпта + поиск топ-5 чанков =====
    with st.spinner("Ищем похожие данные..."):
        query_emb = get_query_embedding(prompt)

    ranked = []
    for chunk in st.session_state.chunks:
        sim = cosine_similarity(query_emb, chunk["embedding"])
        ranked.append((chunk, float(sim)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_chunks = ranked[:5]

    # ===== ШАГ 3: Контекст для модели =====
    context_parts = []
    for i, (chunk, sim) in enumerate(top_chunks):
        context_parts.append(
            f"[Источник: {chunk['source']}]\n"
            f"Чанк #{i + 1} (схожесть: {sim:.4f}):\n{chunk['text']}"
        )
    context = "\n---\n".join(context_parts)

    rag_system = (
        "Ты инженер-технолог КБМ. На основе КОНТЕКСТА рассчитай оборудование.\n"
        "Ответ СТРОГО в формате JSON:\n"
        "{ 'summary': 'текстовый ответ', 'equipment_table': [{'name': 'название', 'qty': число, 'spec': 'характеристика'}] }."
    )

    # ===== ШАГ 4+5+6: Чат, таблица, источники =====
    col_chat, col_result, col_sources = st.columns([1, 1.5, 1])

    # — Колонка 1: Чат —
    with col_chat:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            msg_placeholder = st.empty()
            # === Вызов LLM через HTTP с fallback моделей и детальными ошибками ===
            def safe_ctx(t):
                t = re.sub(r";+", " ", t)
                t = re.sub(r"\s+", " ", t).strip()
                allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.-,:()? /") | set(["'", '"'])
                return "".join(c for c in t if c in allowed)
            
            system_prompt = (
                "Ты инженер-технолог КБМ. Отвечай кратко и по существу, опираясь ТОЛЬКО на предоставленный контекст. "
                "Если информации нет, так и скажи."
            )

            api_key_chat = os.getenv("SCHOOL_API_KEY") or "uii_ai_VnEVVZlPavwIhTry7Ihc9nhPdjDyHVuMcOvYETg4Ktw"
            chat_url = "https://ai-api.neural-university.ru/v1/chat/completions"
            
            # Fallback модели (все реальные модели платформы):
            model_candidates = [
                "gpt-5.6-luna",   # оригинальная модель
                "gpt-6-luna",     # самая эффективная Luna
                "gpt-5.6-terra",  # баланс интеллекта и стоимости
                "gpt-5.6-sol",    # флагман GPT-5.6
                "gpt-6-sol",      # флагман GPT-6
            ]
            
            raw_answer = None
            
            st.caption(f"URL: {chat_url}")
            st.caption(f"Длина контекста: {len(safe_ctx(context))} символов")
            st.caption(f"Промпт пользователя: {prompt}")

            for model_name in model_candidates:
                st.caption(f"Пробуем модель: `{model_name}`")
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Контекст:\n{safe_ctx(context)}\n\nВопрос: {safe_ctx(prompt)}"}
                ]
                
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.0
                }
                
                debug_payload = json.dumps(payload, ensure_ascii=False)[:500]
                st.caption(f"Payload (фрагмент): {debug_payload[:200]}...")

                for attempt in range(3):
                    try:
                        r = requests.post(chat_url, json=payload, headers={"Authorization": f"Bearer {api_key_chat}"}, timeout=60)
                        
                        st.caption(f"Ответ сервера (model={model_name}, attempt {attempt+1}/3): HTTP {r.status_code}, длина: {len(r.text)}")
                        
                        if r.status_code == 200:
                            raw_answer = r.json()["choices"][0]["message"]["content"]
                            st.caption(f"Успех! Модель '{model_name}' вернула {len(raw_answer)} символов")
                            break
                        
                        error_body = ""
                        try:
                            err_json = r.json()
                            error_body = json.dumps(err_json, ensure_ascii=False, indent=2)
                        except Exception:
                            error_body = r.text[:500]
                        
                        if r.status_code == 400:
                            st.error(f"HTTP 400 Bad Request (model={model_name})")
                            st.code(error_body, language="json")
                            st.caption("Шлюз отверг запрос - проверьте название модели или формат.")
                            break
                            
                        if r.status_code == 401:
                            st.error(f"HTTP 401 Unauthorized (model={model_name}): неверный API ключ")
                            st.code(error_body, language="json")
                            break
                            
                        if r.status_code == 404:
                            st.error(f"HTTP 404 Not Found (model={model_name})")
                            st.code(error_body, language="json")
                            break
                            
                        if r.status_code >= 500:
                            wt = (attempt + 1) * 2
                            st.caption(f"Server error ({r.status_code}), retry через {wt}с...")
                            time.sleep(wt)
                        else:
                            st.error(f"Неожиданный HTTP статус {r.status_code} для '{model_name}'")
                            st.code(error_body, language="json")
                            break
                                
                    except requests.exceptions.Timeout as e:
                        st.error(f"Таймаут (model={model_name}, attempt {attempt+1}/3)")
                        time.sleep((attempt + 1) * 2)
                    except requests.exceptions.ConnectionError as e:
                        st.error(f"Ошибка соединения (model={model_name}): {e}")
                        time.sleep((attempt + 1) * 2)
                    except Exception as e:
                        st.error(f"Ошибка API (model={model_name}): {type(e).__name__}: {e}")
                        st.exception(e)
                        break
                
                if raw_answer:
                    break
                st.caption(f"Модель '{model_name}' не сработала, пробуем следующую...")

            if not raw_answer:
                msg_placeholder.error(f"Не удалось получить ответ. Попробованы: {', '.join(model_candidates)}. См. логи выше.")
                st.stop()


            # Отладочный вывод — сырой ответ модели
            st.expander("🔍 Сырой ответ модели (для отладки)", expanded=True).code(raw_answer)

            # Парсинг JSON
            summary_text = raw_answer
            equipment_data = []

            try:
                # Пробуем извлечь блок ```json ... ```
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
                    summary_text = ""
                
                st.caption(f"✅ JSON распарсен успешно! Полей найдено: {len(equipment_data)}")
            except json.JSONDecodeError as e:
                st.error(f"❌ Ошибка парсинга JSON: {e}")
                st.caption("Модель вернула некорректный JSON. См. сырой ответ выше в expander'е.")
            except Exception as e:
                st.error(f"❌ Неожиданная ошибка при разборе ответа: {type(e).__name__}: {e}")
                st.exception(e)

            msg_placeholder.markdown(summary_text if summary_text else raw_answer)

    # — Колонка 2: Таблица результата —
    with col_result:
        st.subheader("📊 Результат расчета")
        if equipment_data:
            table_rows = []
            for item in equipment_data:
                if isinstance(item, dict):
                    table_rows.append({
                        "Название": str(item.get("name", "")),
                        "Кол-во": str(item.get("qty", "")),
                        "Характеристика": str(item.get("spec", "")),
                    })
            if table_rows:
                df_equipment = pd.DataFrame(table_rows)
                st.dataframe(df_equipment, use_container_width=True, hide_index=True)
            else:
                st.info("Нет данных об оборудовании.")
        else:
            st.info("Структурированные данные не получены.")

    # — Колонка 3: Источники —
    with col_sources:
        st.subheader("📚 Источники")
        with st.expander("Показать 5 найденных фрагментов"):
            for i, (chunk, sim) in enumerate(top_chunks):
                st.caption(f"#{i + 1} · схожесть: {sim:.4f}")
                truncated = chunk["text"][:400]
                if len(chunk["text"]) > 400:
                    truncated += "..."
                st.text_area("", value=truncated, height=200, disabled=True)
