"""
Data Cleaning Module
Handles missing values, duplicates, and basic type fixes for uploaded datasets.
"""

import pandas as pd


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Inspect the dataframe and return a summary of quality issues
    before any cleaning is applied.
    """
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_types": df.dtypes.astype(str).to_dict(),
    }
    return report


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply automated cleaning steps:
    - Drop exact duplicate rows
    - Fill missing numeric values with the column median
    - Fill missing text values with 'Unknown'
    - Attempt to parse date-like columns

    Returns the cleaned dataframe and a summary of what was changed.
    """
    original_rows = len(df)
    cleaned_df = df.copy()

    # 1. Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = original_rows - len(cleaned_df)

    # 2. Handle missing values column by column
    filled_numeric_cols = []
    filled_text_cols = []

    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
            median_value = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_value)
            filled_numeric_cols.append(col)
        else:
            cleaned_df[col] = cleaned_df[col].fillna("Unknown")
            filled_text_cols.append(col)

    # 3. Try to auto-detect and parse date columns (by column name hint)
    date_like_cols = []
    for col in cleaned_df.columns:
        if any(keyword in col.lower() for keyword in ["date", "time", "created", "updated"]):
            try:
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors="coerce")
                date_like_cols.append(col)
            except Exception:
                pass

    summary = {
        "duplicates_removed": duplicates_removed,
        "numeric_columns_filled": filled_numeric_cols,
        "text_columns_filled": filled_text_cols,
        "date_columns_parsed": date_like_cols,
        "final_row_count": len(cleaned_df),
    }

    return cleaned_df, summary