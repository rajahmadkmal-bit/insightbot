"""
Visualization Module
Detects column types and automatically generates the most suitable chart.
"""

import pandas as pd
import plotly.express as px


def classify_columns(df: pd.DataFrame) -> dict:
    """
    Classify each column as 'numeric', 'date', or 'categorical'.
    """
    classification = {"numeric": [], "date": [], "categorical": []}

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            classification["date"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            classification["numeric"].append(col)
        else:
            classification["categorical"].append(col)

    return classification


def generate_auto_charts(df: pd.DataFrame, max_charts: int = 4) -> list:
    """
    Generate a list of Plotly figures chosen automatically based on
    the data's column types. Returns a list of (title, figure) tuples.
    """
    cols = classify_columns(df)
    charts = []

    # Date + numeric -> line chart (trend over time)
    if cols["date"] and cols["numeric"]:
        date_col = cols["date"][0]
        for num_col in cols["numeric"][:2]:
            fig = px.line(
                df.sort_values(date_col),
                x=date_col,
                y=num_col,
                title=f"{num_col} over time",
            )
            charts.append((f"{num_col} Trend", fig))

    # Categorical + numeric -> bar chart (comparison across categories)
    if cols["categorical"] and cols["numeric"]:
        cat_col = cols["categorical"][0]
        num_col = cols["numeric"][0]
        grouped = df.groupby(cat_col)[num_col].sum().reset_index()
        grouped = grouped.sort_values(num_col, ascending=False).head(10)
        fig = px.bar(
            grouped,
            x=cat_col,
            y=num_col,
            title=f"{num_col} by {cat_col}",
        )
        charts.append((f"{num_col} by {cat_col}", fig))

    # Single categorical -> pie chart (proportion breakdown)
    if cols["categorical"]:
        cat_col = cols["categorical"][0]
        value_counts = df[cat_col].value_counts().head(8).reset_index()
        value_counts.columns = [cat_col, "count"]
        fig = px.pie(
            value_counts,
            names=cat_col,
            values="count",
            title=f"Distribution of {cat_col}",
        )
        charts.append((f"{cat_col} Distribution", fig))

    # Two numeric columns -> scatter plot (relationship)
    if len(cols["numeric"]) >= 2:
        x_col, y_col = cols["numeric"][0], cols["numeric"][1]
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        charts.append((f"{y_col} vs {x_col}", fig))

    return charts[:max_charts]