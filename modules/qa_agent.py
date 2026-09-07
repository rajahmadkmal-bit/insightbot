"""
Natural Language Q&A Agent
Lets the user ask questions about their data in plain language.
The agent writes and executes pandas code behind the scenes.
"""

import os
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


def create_qa_agent(df: pd.DataFrame):
    """
    Create a LangChain agent that can answer natural language
    questions about the given dataframe.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=api_key,
        temperature=0,
    )

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,  # required by langchain-experimental; runs locally on your own data
    )
    return agent


def ask_question(agent, question: str) -> str:
    """
    Ask a question to the Q&A agent and return its answer.
    """
    try:
        response = agent.invoke(question)
        return response.get("output", str(response))
    except Exception as e:
        return f"Sorry, I couldn't process that question. Error: {str(e)}"