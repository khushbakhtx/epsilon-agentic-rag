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

try:
    df = pd.read_csv("data/all_data.csv")
except FileNotFoundError:
    st.error("CSV file 'data/all_data.csv' not found.")
    st.stop()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0
)

agent_executor = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type="tool-calling",
    verbose=True,
    allow_dangerous_code=True
)

st.title("epsilon.ai agent")
st.write("Ask questions about the data in 'all_data.csv'. Type your question below.")

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