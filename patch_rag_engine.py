#!/usr/bin/env python3
"""Патч rag_engine.py для совместимости с pymorphy2 + Python 3.11+"""
import sys

path = "/Users/andrejabramov/Desktop/Urban/Urban/kbm_ai_technology/rag_engine.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Патч 1: pymorphy2 compat (добавляем перед import pymorphy2)
old_imports = """from rank_bm25 import BM25Okapi
from pymorphy2 import MorphAnalyzer"""

new_imports = """from rank_bm25 import BM25Okapi

# ---------- Patch pymorphy2 для Python 3.11+ ----------
import inspect as _inspect
if not hasattr(_inspect, "getargspec"):
    def _getargspec_compat(func):
        return _inspect.ArgSpec(
            parameters=_inspect.getfullargspec(func).args,
            varargs=_inspect.getfullargspec(func).varargs,
            keywords=_inspect.getfullargspec(func).varkw,
            defaults=_inspect.getfullargspec(func).defaults,
        )
    _inspect.getargspec = _getargspec_compat

from pymorphy2 import MorphAnalyzer"""

content = content.replace(old_imports, new_imports)

# Патч 2: NLTK punkt - пробуем разные имена
old_nltk = """for resource in ["stopwords", "tokenizers/punkt_tab"]:
    try:
        nltk.download(resource, download_dir=NLTK_DATA_DIR, quiet=True)
        nltk.data.find(f"corpora/{resource}" if "stopwords" in resource else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)"""

new_nltk = '''# Загружаем стоп-слова
nltk.download("stopwords", download_dir=NLTK_DATA_DIR, quiet=True)
nltk.data.find("corpora/stopwords")

# Загружаем punkt (пробуем punkt_tab, потом punkt)
punkt_loaded = False
for punkt_name in ("punkt_tab", "punkt"):
    try:
        nltk.download(f"tokenizers/{punkt_name}", download_dir=NLTK_DATA_DIR, quiet=True)
        nltk.data.find(f"tokenizers/{punkt_name}")
        punkt_loaded = True
        break
    except LookupError:
        pass'''

content = content.replace(old_nltk, new_nltk)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"✅ {path} патчнут успешно")
