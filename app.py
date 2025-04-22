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
    "data/forecast_data.csv",
    "data/oper_expenses.csv",
    "data/test_metamodels_metrics.csv",
    "data/all_correlations.csv",
    "data/all_data.csv"
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
Ты ИИ агент, который может выполнять команды и отвечать на вопросы о данных в CSV файлах:
У тебя есть доступ к следующим CSV файлам:
{dataframe_info}
Когда отвечаешь на вопрос, используй данные из CSV файлов, но ответ должен быть один, не по всем данным а только по релевантным.
"""

dataframe_info = "\n".join(
    f"- {name}: Contains data about {name.replace('_', ' ')}." for name, df in zip(dataframe_names, dataframes)
)

agent_executor = create_pandas_dataframe_agent(
    llm,
    dataframes,
    agent_type="tool-calling",
    verbose=True,
    allow_dangerous_code=True,
    prefix=system_prompt.format(dataframe_info=dataframe_info)
)

st.title("epsilon.ai Agent")
st.write("Можно задавать вопросы о данных в all_correlations.csv, all_data.csv, forecast_data.csv, oper_expenses.csv и test_metamodels_metrics.csv")

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