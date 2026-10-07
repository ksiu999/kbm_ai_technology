import os
import subprocess
import io
import streamlit as st
import requests
from dotenv import load_dotenv
from rag_engine import HybridSearchEngine
from config import (
    ROUTER_BASE_URL,
    ROUTER_MODEL,
    SCHOOL_BASE_URL,
    SCHOOL_EMBEDDING_MODEL,
)


# --- Загрузка секретов из .env.sops с надежным фоллбэком ---
def load_sops_env(file_path: str = ".env.sops"):
    """Расшифровывает .env.sops и загружает переменные в окружение."""
    sops_success = False
    if os.path.exists(file_path):
        try:
            result = subprocess.run(
                ["sops", "-d", file_path], capture_output=True, text=True, check=True
            )
            load_dotenv(stream=io.StringIO(result.stdout))
            print("✅ Секреты успешно загружены из .env.sops")
            sops_success = True
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Ошибка расшифровки .env.sops: {e.stderr.strip()}")
        except FileNotFoundError:
            print("⚠️ Утилита 'sops' не найдена в PATH.")

    # Если SOPS не сработал или файла нет, грузим обычный .env
    if not sops_success:
        load_dotenv()
        print("ℹ️ Загружены переменные из стандартного файла .env")

    # Мгновенная проверка, что ключи действительно загрузились
    if not os.getenv("SCHOOL_API_KEY"):
        print(
            "❌ КРИТИЧЕСКАЯ ОШИБКА: SCHOOL_API_KEY пустой! Проверьте содержимое файла .env"
        )
    else:
        print("✅ SCHOOL_API_KEY успешно загружен и готов к работе.")

    if not os.getenv("ROUTER_API_KEY"):
        print(
            "❌ КРИТИЧЕСКАЯ ОШИБКА: ROUTER_API_KEY пустой! Проверьте содержимое файла .env"
        )
    else:
        print("✅ ROUTER_API_KEY успешно загружен и готов к работе.")


# Вызов функции ДО любого использования os.getenv()
load_sops_env()

# --- Константы (приоритет: .env.sops -> config.py) ---
DILAB_DB_PATH = "db/dilab_chunks.jsonl"

# URL для чата (по умолчанию из config.py, но можно переопределить в .env.sops)
CHAT_BASE_URL = os.getenv("ROUTER_BASE_URL", ROUTER_BASE_URL)
CHAT_URL = f"{CHAT_BASE_URL}/chat/completions"

# Модель для чата (по умолчанию из config.py)
CHAT_MODEL = os.getenv("ROUTER_MODEL", ROUTER_MODEL)

# URL для эмбеддингов (по умолчанию из config.py)
EMBED_URL = os.getenv("EMBED_URL", f"{SCHOOL_BASE_URL}/embeddings")

# --- Проверка API ключей для UI ---
school_key = os.getenv("SCHOOL_API_KEY")
router_key = os.getenv("ROUTER_API_KEY")

if not school_key:
    st.error("❌ SCHOOL_API_KEY не найден в окружении")
else:
    st.sidebar.success("✅ SCHOOL_API_KEY загружен")

if not router_key:
    st.error("❌ ROUTER_API_KEY не найден в окружении")
else:
    st.sidebar.success("✅ ROUTER_API_KEY загружен")


# --- Инициализация движка с кэшированием ---
@st.cache_resource(show_spinner="Загрузка индекса и инициализация BM25...")
def init_engine():
    return HybridSearchEngine(
        db_path=DILAB_DB_PATH, embed_url=EMBED_URL, api_key=school_key, alpha=0.5
    )


st.title("🏭 ДиЛаб — Агент по должностным инструкциям")

try:
    engine = init_engine()
    has_index = len(engine.chunks) > 0
    st.sidebar.success(f"📚 База знаний ДиЛаб: {len(engine.chunks)} чанков")
except FileNotFoundError:
    has_index = False
    st.warning(
        f"Индекс не найден: {DILAB_DB_PATH}. Запустите `python ingest.py` для создания базы."
    )
    engine = None

prompt = st.chat_input("Задайте вопрос по инструкциям...", disabled=not has_index)

if prompt and has_index and engine:
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Выполняю гибридный поиск и генерирую ответ..."):
            # 1. Поиск
            search_result = engine.search(prompt, top_k=5)

            # 2. UI: Логика работы RAG
            with st.expander("🔍 Логика работы RAG (Hybrid Search)", expanded=False):
                st.write(
                    f"**Извлеченные ключевые слова:** `{', '.join(search_result['keywords'])}`"
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Топ-5 Embedding:**")
                    for item in search_result["top_embedding"][:5]:
                        st.caption(
                            f"Скор: {item['embed_score']:.4f} | {item['source']}"
                        )
                with col2:
                    st.markdown("**Топ чанков по ключевым словам (BM25):**")
                    for item in search_result["top_bm25"]:
                        st.caption(f"Скор: {item['bm25_score']:.4f} | {item['source']}")

                st.markdown("**Финальный ранжированный список (передается в LLM):**")
                for item in search_result["final_results"]:
                    st.markdown(
                        f"- **{item['source']}** (Итоговый скор: {item['final_score']:.4f})\n  > {item['text'][:150]}..."
                    )

            # 3. Формирование контекста и запрос к LLM
            context = "\n\n".join(
                [
                    f"Источник: {c['source']}\nТекст: {c['text']}"
                    for c in search_result["final_results"]
                ]
            )

            system_prompt = "Ты полезный ассистент. Отвечай строго на основе предоставленного контекста. Если ответа нет в контексте, скажи об этом."
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Контекст:\n{context}\n\nВопрос пользователя: {prompt}",
                },
            ]

            headers = {
                "Authorization": f"Bearer {router_key}",
                "Content-Type": "application/json",
            }

            # УЛУЧШЕННАЯ ОТЛАДКА: покажем длину ключа, чтобы исключить его обрезку при расшифровке SOPS
            key_len = len(router_key) if router_key else 0
            key_start = router_key[:10] if key_len > 10 else router_key
            print(f"🔍 DEBUG: Запрос на {CHAT_URL}")
            print(f"🔍 DEBUG: Модель: {CHAT_MODEL}")
            print(f"🔍 DEBUG: Ключ (длина={key_len}): {key_start}...")

            payload = {"model": CHAT_MODEL, "messages": messages}

            try:
                response = requests.post(CHAT_URL, json=payload, headers=headers)
                response.raise_for_status()
                ai_answer = response.json()["choices"][0]["message"]["content"]
                st.write(ai_answer)
            except Exception as e:
                st.error(f"Ошибка при запросе к RouterAI: {e}")

elif not has_index and prompt:
    st.error("База данных пуста. Сначала запустите индексацию.")
