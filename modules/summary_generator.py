"""
Executive Summary Generator
Sends key data statistics to the Gemini LLM (free tier) and returns
a plain-language business summary.
"""

import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """
    Initialize the free Gemini model.
    Requires GEMINI_API_KEY to be set in the .env file.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

    return ChatGoogleGenerativeAI(
       model="gemini-flash-latest",
        google_api_key=api_key,
        temperature=0.3,
    )


def build_stats_context(df: pd.DataFrame) -> str:
    """
    Build a compact text summary of the dataframe's key statistics
    to feed into the LLM prompt (keeps token usage low).
    """
    numeric_summary = df.describe(include="number").to_string()
    top_rows_preview = df.head(5).to_string()

    context = (
        f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
        f"Column names: {', '.join(df.columns)}\n\n"
        f"Numeric summary statistics:\n{numeric_summary}\n\n"
        f"Sample rows:\n{top_rows_preview}"
    )
    return context


def generate_executive_summary(df: pd.DataFrame, language: str = "English") -> str:
    """
    Generate a business-friendly executive summary from the dataframe.
    language: 'English' or 'Urdu' — controls the output language.
    """
    llm = get_llm()
    stats_context = build_stats_context(df)

    prompt = f"""
You are a business intelligence analyst. Based on the following dataset summary,
write a short executive summary (5-7 sentences) for a business owner who has
no technical background.

Focus on:
- What the data shows overall
- Where performance looks strong
- Where there might be problems or risks
- One or two actionable recommendations

Write the summary in {language}. Keep it clear and non-technical.

Dataset summary:
{stats_context}
"""

    response = llm.invoke(prompt)
    return response.content