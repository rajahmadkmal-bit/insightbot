"""
InsightBot - Automated BI & Insights Agent
Main Streamlit application entry point.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from modules.data_cleaning import get_data_quality_report, clean_dataframe
from modules.visualization import generate_auto_charts
from modules.summary_generator import generate_executive_summary
from modules.qa_agent import create_qa_agent, ask_question

st.set_page_config(page_title="InsightBot - AI Data Analyst", layout="wide")

st.title("InsightBot")
st.caption("Your automated AI Business Intelligence & Insights agent")

# --- Session state setup ---
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "qa_agent" not in st.session_state:
    st.session_state.qa_agent = None

# --- Step 1: File upload ---
uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head())

    # --- Step 2: Data quality report + cleaning ---
    st.subheader("Data Quality Report")
    quality_report = get_data_quality_report(raw_df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", quality_report["total_rows"])
    col2.metric("Total Columns", quality_report["total_columns"])
    col3.metric("Duplicate Rows", quality_report["duplicate_rows"])

    if st.button("Clean Data Automatically"):
        cleaned_df, clean_summary = clean_dataframe(raw_df)
        st.session_state.cleaned_df = cleaned_df

        st.success("Data cleaned successfully.")
        st.json(clean_summary)

if st.session_state.cleaned_df is not None:
    df = st.session_state.cleaned_df

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head())

    # --- Step 3: Auto visualizations ---
    st.subheader("Automated Charts")
    charts = generate_auto_charts(df)
    for title, fig in charts:
        st.plotly_chart(fig, use_container_width=True)

    # --- Step 4: Executive summary ---
    st.subheader("Executive Summary")
    summary_language = st.radio("Summary language", ["English", "Urdu"], horizontal=True)
    if st.button("Generate Executive Summary"):
        with st.spinner("Analyzing data..."):
            summary = generate_executive_summary(df, language=summary_language)
            st.write(summary)

    # --- Step 5: Natural language Q&A ---
    st.subheader("Ask Questions About Your Data")
    if st.session_state.qa_agent is None:
        st.session_state.qa_agent = create_qa_agent(df)

    user_question = st.text_input("Ask a question, e.g. 'Which product sold the most last month?'")
    if user_question:
        with st.spinner("Thinking..."):
            answer = ask_question(st.session_state.qa_agent, user_question)
            st.write(answer)