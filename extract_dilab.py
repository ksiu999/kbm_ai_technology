#!/usr/bin/env python3
"""
Скрипт-оркестратор: извлечение текста из .docx файлов базы знаний "ДиЛаб".
Результат дозаписывается в docs/dilab_base.txt.
Версия 2.0 — корректная обработка soft returns (слова не слипаются).
"""
import os
import re
from docx import Document

# Пути
DILAB_DIR = "docs/dilab"
OUTPUT_FILE = "docs/dilab_base.txt"


def clean_and_split(text: str) -> list[str]:
    """
    Очищает текст из параграфа docx:
    - Заменяет soft returns (\\n) на пробелы (чтобы слова не слипались)
    - Удаляет запрещённые символы
    - Разбивает на строки по двойным переносам или маркированным спискам
    - Схлопывает множественные пробелы
    """
    # Soft returns -> пробел
    text = text.replace('\x0c', '\n')  # form feed -> newline
    text = text.replace('\r', '')      # carriage return
    
    # Сохраняем маркеры списков для разбивки
    # Разделяем по маркированным/нумерованным спискам, которые часто идут с \n
    lines = text.split('\n')
    
    blocks = []
    for line in lines:
        # Удаляем запрещённые символы (кроме букв, цифр, пробелов, базовой пунктуации)
        cleaned = re.sub(
            r'[^\w\s.,\-:()!?\'\"/@#$%^&*+=~`<>|\[\]{};№%—…]',
            '', line
        )
        # Схлопываем множественные пробелы
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
        
        if len(cleaned) > 0:
            blocks.append(cleaned)
    
    return blocks


def extract_text_from_docx(filepath: str) -> list[str]:
    """Извлечь текстовые параграфы из .docx файла."""
    doc = Document(filepath)
    paragraphs = []
    
    # Параграфы
    for para in doc.paragraphs:
        blocks = clean_and_split(para.text)
        paragraphs.extend(blocks)
    
    # Таблицы — текст в ячейках часто не попадает в paragraphs
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    blocks = clean_and_split(para.text)
                    paragraphs.extend(blocks)
    
    return paragraphs


def main():
    docx_files = sorted([
        f for f in os.listdir(DILAB_DIR) if f.lower().endswith('.docx')
    ])

    print(f"📂 Найдено {len(docx_files)} файлов .docx в папке '{DILAB_DIR}'\n")

    if len(docx_files) == 0:
        print("❌ Файлы не найдены! Проверьте путь.")
        return

    total_chars = 0
    total_success = 0
    total_errors = 0
    failed_files = []

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for i, filename in enumerate(docx_files, start=1):
            filepath = os.path.join(DILAB_DIR, filename)
            try:
                print(f"[{i}/{len(docx_files)}] Обработка: {filename} ... ", end="", flush=True)
                
                paragraphs = extract_text_from_docx(filepath)
                
                if not paragraphs:
                    print(f"⚠️ Пустой документ (пропуск)")
                    total_errors += 1
                    failed_files.append(filename)
                    continue

                separator = f"\n\n=== НАЧАЛО ДОКУМЕНТА: {filename} ===\n\n"
                out.write(separator)

                file_chars = 0
                for para in paragraphs:
                    out.write(para + "\n")
                    file_chars += len(para) + 1

                total_chars += file_chars
                total_success += 1
                print(f"✅ ({len(paragraphs)} блоков, ~{file_chars} символов)")

            except Exception as e:
                print(f"❌ Ошибка: {type(e).__name__}: {str(e)[:80]}")
                total_errors += 1
                failed_files.append(filename)

    print("\n" + "=" * 60)
    print(f"🎉 Готово!")
    print(f"   Обработано файлов:      {total_success}")
    print(f"   Ошибок/пропусков:       {total_errors}")
    print(f"   Итого символов:         {total_chars}")
    print(f"   Итоговый файл:          {OUTPUT_FILE}")

    if failed_files:
        print(f"\n⚠️ Пропущенные файлы:")
        for f in failed_files:
            print(f"   - {f}")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    print(f"   Итого строк в файле:    {line_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
