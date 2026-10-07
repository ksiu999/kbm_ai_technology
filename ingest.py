import os
import subprocess
import io
import jsonlines
import requests
from docx import Document
from dotenv import load_dotenv


# --- Загрузка секретов из .env.sops ---
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

    # Мгновенная проверка, что ключ действительно загрузился
    if not os.getenv("SCHOOL_API_KEY"):
        print(
            "❌ КРИТИЧЕСКАЯ ОШИБКА: SCHOOL_API_KEY пустой! Проверьте содержимое файла .env"
        )
    else:
        print("✅ SCHOOL_API_KEY успешно загружен и готов к работе.")


# Вызов функции ДО любого использования os.getenv()
load_sops_env()


DOCS_DIR = "docs/dilab"
DB_PATH = "db/dilab_chunks.jsonl"
EMBED_URL = os.getenv("EMBED_URL", "https://ai-api.neural-university.ru/v1/embeddings")
API_KEY = os.getenv("SCHOOL_API_KEY", "")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_embedding(text: str) -> list[float]:
    """Получение эмбеддинга через API."""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": text}
    response = requests.post(EMBED_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def chunk_text(text: str, source: str) -> list[dict]:
    """Разбиение текста на чанки с перекрытием."""
    chunks = []
    start = 0
    text_len = len(text)
    chunk_id = 0

    while start < text_len:
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        chunks.append(
            {
                "id": chunk_id,
                "text": chunk_text.strip(),
                "embedding": [],  # Будет заполнено ниже
                "source": source,
            }
        )
        chunk_id += 1
        start = end - CHUNK_OVERLAP

    return chunks


def main():
    os.makedirs("db", exist_ok=True)

    if not os.path.exists(DOCS_DIR):
        print(f"❌ Директория {DOCS_DIR} не найдена.")
        return

    all_chunks = []
    docx_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".docx")]

    print(f"Найдено {len(docx_files)} файлов .docx. Начинаем обработку...")

    for filename in docx_files:
        filepath = os.path.join(DOCS_DIR, filename)
        doc = Document(filepath)
        full_text = "\n".join(
            [para.text for para in doc.paragraphs if para.text.strip()]
        )

        chunks = chunk_text(full_text, filename)
        print(f"  Обработан {filename}: {len(chunks)} чанков.")

        for chunk in chunks:
            try:
                chunk["embedding"] = get_embedding(chunk["text"])
                all_chunks.append(chunk)
            except Exception as e:
                print(f"  ⚠️ Ошибка эмбеддинга для чанка из {filename}: {e}")

    with jsonlines.open(DB_PATH, mode="w") as writer:
        writer.write_all(all_chunks)

    print(f"✅ Успешно сохранено {len(all_chunks)} чанков в {DB_PATH}")


if __name__ == "__main__":
    main()
