import pandas as pd
import numpy as np


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Return a comprehensive profile of a DataFrame."""
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    missing = {
        col: {"count": int(missing_counts[col]), "pct": float(missing_pct[col])}
        for col in df.columns
    }

    numeric_cols = df.select_dtypes(include="number").columns
    numeric_summary = (
        df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
    )

    return {
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": missing,
        "duplicates": int(df.duplicated().sum()),
        "memory_usage": int(df.memory_usage(deep=True).sum()),
        "numeric_summary": numeric_summary,
        "column_list": df.columns.tolist(),
    }


def get_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a boolean mask where True indicates an IQR outlier."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (df[col] < lower) | (df[col] > upper)


def get_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3) -> pd.Series:
    """Return a boolean mask where True indicates a z-score outlier."""
    z = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    return z.abs() > threshold
