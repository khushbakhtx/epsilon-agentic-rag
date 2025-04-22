import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

csv_files = [
    "data/full_data.csv",
    ]
dataframes = []
dataframe_names = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(f"{csv_file}")
        dataframes.append(df)
        dataframe_names.append(os.path.basename(csv_file).replace(".csv", ""))
    except FileNotFoundError:
        st.error(f"CSV file '{csv_file}' not found.")
        st.stop()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0
)

system_prompt = """
Ты - ИИ Агент - эксперт по анализу структурированных CSV данных. Твоя задача — отвечать на вопросы пользователя, используя данные из следующих файлов:

1. **full_data**: Основной файл, который включает данные по выручке, затратам, прибыли и другим ключевым метрикам для разных подразделений компании с учетом даты.
   - Колонки:
     - **Дата**: Дата записи в формате YYYY-MM-DD год-месяц(1-январь, 12-декабрь)-день.
     - **Подразделение компании (дивизион)**: Название подразделения компании.
     - **Доход(сколько заработали)**: Выручка за период, в тенге.
     - **Расход(сколько потратили за период)**: Все затраты за период, в тенге.
     - **Валовая_прибыль(Доход - расход)**: Прибыль до вычета операционных расходов, в тенге (рассчитывается как Доход - Расход).
     - **EBITDA**: Прибыль до вычета налогов, процентов и амортизации, в тенге.
     - **Отток клиентов(сколько ушло)**: Число ушедших клиентов за определенный период времени с 2023 по 2025 год.
     - **Приток клиентов (сколько клиентов пришло)**: Число новых клиентов с 2023 по 2025 год.
     - **ARPU**: Средний доход на одного клиента, в тенге.
     - **операционные_расходы_млн_тенге**: Расходы на операционную деятельность, в миллионах тенге.
     - **операционная_прибыль_млн_тенге**: Прибыль после вычета операционных расходов, в миллионах тенге.

Ответь на вопросы пользователя, используя предоставленные данные.
Если пользователь просто хочет узнать о содержимом, то передай ему эту информацию из данных (не считая ничего).
- **Не создавай новые данные, не придумывай значения и не генерируй синтетические DataFrame, и не суммируй ничего пока пользователь прямо не попросит**.
- Для вопросов о конкретном месяце (например, декабрь 2024), фильтруй к примеру вот так pd.to_datetime(df["Дата"]).dt.to_period("M") == "YYYY-MM".
"""

agent_executor = create_pandas_dataframe_agent(
    llm,
    dataframes,
    agent_type="tool-calling",
    verbose=True,
    allow_dangerous_code=True,
    prefix=system_prompt
)

st.title("epsilon.ai Agent")
st.write("Можно задавать вопросы о данных в CSV файлах.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Your question:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        with st.spinner("Processing..."):
            response = agent_executor.invoke({"input": user_input})
            output = response["output"]
        
        st.session_state.messages.append({"role": "assistant", "content": output})
        with st.chat_message("assistant"):
            st.markdown(output)
    except Exception as e:
        error_message = f"Error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.markdown(error_message)