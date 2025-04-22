from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd

load_dotenv()

def main():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set")
        return

    st.set_page_config(page_title="epsilon.ai")
    st.header("Epsilon Agentic RAG")

    csv_files = [
        "data/all_correlations.csv",
        "data/all_data.csv",
        "data/forecast_data.csv",
        "data/oper_expenses.csv"
    ]

    custom_prompt_template = PromptTemplate(
        input_variables=["input", "file_name"],
        template="""
        You are a data analyst working with a CSV file named {file_name}. Your task is to answer questions about the data in a clear, concise, and accurate manner. Use Python and pandas to analyze the data. Follow these guidelines:
        - Provide answers in a structured format, using bullet points where applicable.
        - Answer in language that user asks a question. If it is RUSSIAN, answer in RUSSIAN. If it is ENGLISH, answer in ENGLISH.

        File: {file_name}
        User question: {input}

        Answer the question based on the data in {file_name}.
        """
    )

    agents = []
    file_summaries = []
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            st.warning(f"File {csv_file} not found. Skipping.")
            continue

        agent = create_csv_agent(
            OpenAI(temperature=0),
            csv_file,
            verbose=True,
            allow_dangerous_code=True
        )
        agents.append((csv_file, agent))

        try:
            df = pd.read_csv(csv_file)
            summary = {
                "name": csv_file,
                "columns": df.columns.tolist(),
                "row_count": len(df)
            }
            file_summaries.append(summary)
        except Exception as e:
            st.warning(f"Error reading {csv_file}: {str(e)}")

    if file_summaries:
        st.write("### Коротко о данных")
        for summary in file_summaries:
            st.write(f"**{summary['name']}**: {summary['row_count']} строк, Колонки: {', '.join(summary['columns'])}")
    else:
        st.error("No valid CSV files found.")
        return

    user_question = st.text_input("Задайтк вопрос по данным: ")
    if user_question:
        with st.spinner("Processing..."):
            response = ""
            for file_name, agent in agents:
                try:
                    result = agent.run(custom_prompt_template.format(input=user_question, file_name=file_name))
                    response += f"**{file_name}**: {result}\n"
                except Exception as e:
                    response += f"**{file_name}**: Error processing question ({str(e)})\n"
            st.write(response)

if __name__ == "__main__":
    main()