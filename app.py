import os
import streamlit as st
import openai
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# --- Проверка API ключа ---
api_key = os.getenv("SCHOOL_API_KEY")
if not api_key:
    st.error("API ключ не найден")
else:
    st.sidebar.success("✅ API ключ загружен успешно")

# --- Инициализация клиента OpenAI ---
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://ai-api.neural-university.ru/v1"
)

# --- UI ---
st.title("🏭 Агент расчета оборудования КБМ")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите kbm_base.csv", type=["csv"])

# Чат
prompt = st.chat_input("Задайте вопрос технологу...")

if prompt:
    # Добавляем сообщение пользователя в чат
    with st.chat_message("user"):
        st.write(prompt)

    # Делаем запрос к API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=[
                {"role": "system", "content": "Ты помощник инженера КБМ"},
                {"role": "user", "content": prompt},
            ]
        )
        answer = response.choices[0].message.content
        message_placeholder.write(answer)