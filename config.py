"""
Публичные настройки приложения (не содержат секретов).
Этот файл версионируется в Git.
"""

# RouterAI / OpenRouter настройки
ROUTER_BASE_URL = "https://routerai.ru/api/v1"
ROUTER_MODEL = "qwen/qwen3.7-flash"

# Школьный API (для эмбеддингов)
SCHOOL_BASE_URL = "https://ai-api.neural-university.ru/v1"
SCHOOL_EMBEDDING_MODEL = "text-embedding-3-small"

# SOPS / Age (публичный ключ для шифрования)
AGE_PUBLIC_KEY = "age1e306uw9fgq8mxmxuvxakyj7dzxpstmwh8v3xs6mht588exfkxqpsnrrp8m"

# Streamlit настройки
STREAMLIT_PORT = 8501
STREAMLIT_HOST = "0.0.0.0"
