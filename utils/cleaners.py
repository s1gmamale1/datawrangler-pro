import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def drop_missing_rows(df: pd.DataFrame, subset=None):
    new_df = df.dropna(subset=subset).reset_index(drop=True)
    dropped = len(df) - len(new_df)
    return new_df, {
        "operation": "drop_missing_rows",
        "column": subset or "all",
        "params": {"subset": subset, "rows_dropped": dropped},
        "timestamp": _now(),
    }


def drop_missing_cols(df: pd.DataFrame, threshold: float = 0.5):
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    new_df = df.drop(columns=cols_to_drop)
    return new_df, {
        "operation": "drop_missing_cols",
        "column": cols_to_drop,
        "params": {"threshold": threshold, "cols_dropped": cols_to_drop},
        "timestamp": _now(),
    }


def fill_missing(df: pd.DataFrame, col: str, strategy: str, fill_value=None):
    new_df = df.copy()
    if strategy == "constant":
        new_df[col] = new_df[col].fillna(fill_value)
    elif strategy == "mean":
        new_df[col] = new_df[col].fillna(new_df[col].mean())
    elif strategy == "median":
        new_df[col] = new_df[col].fillna(new_df[col].median())
    elif strategy == "mode":
        mode = new_df[col].mode()
        new_df[col] = new_df[col].fillna(mode[0] if not mode.empty else np.nan)
    elif strategy == "ffill":
        new_df[col] = new_df[col].ffill()
    elif strategy == "bfill":
        new_df[col] = new_df[col].bfill()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return new_df, {
        "operation": "fill_missing",
        "column": col,
        "params": {"strategy": strategy, "fill_value": fill_value},
        "timestamp": _now(),
    }


def drop_duplicates(df: pd.DataFrame, subset=None, keep: str = "first"):
    new_df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    dropped = len(df) - len(new_df)
    return new_df, {
        "operation": "drop_duplicates",
        "column": subset or "all",
        "params": {"subset": subset, "keep": keep, "rows_dropped": dropped},
        "timestamp": _now(),
    }


def convert_type(df: pd.DataFrame, col: str, target_type: str, datetime_format=None):
    new_df = df.copy()
    if target_type in ("numeric", "float", "integer", "int"):
        if new_df[col].dtype == object:
            new_df[col] = (
                new_df[col]
                .astype(str)
                .str.replace(r"[\$,\s]", "", regex=True)
            )
        new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
        if target_type in ("integer", "int"):
            new_df[col] = new_df[col].astype("Int64")
    elif target_type == "category":
        new_df[col] = new_df[col].astype("category")
    elif target_type == "datetime":
        new_df[col] = pd.to_datetime(new_df[col], format=datetime_format, errors="coerce")
    elif target_type == "string":
        new_df[col] = new_df[col].astype(str)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    return new_df, {
        "operation": "convert_type",
        "column": col,
        "params": {"target_type": target_type, "datetime_format": datetime_format},
        "timestamp": _now(),
    }


def standardize_categorical(df: pd.DataFrame, col: str, operations: list):
    new_df = df.copy()
    series = new_df[col].astype(str)
    for op in operations:
        if op == "trim":
            series = series.str.strip()
        elif op == "lower":
            series = series.str.lower()
        elif op == "upper":
            series = series.str.upper()
        elif op == "title":
            series = series.str.title()
        else:
            raise ValueError(f"Unknown operation: {op}")
    new_df[col] = series
    return new_df, {
        "operation": "standardize_categorical",
        "column": col,
        "params": {"operations": operations},
        "timestamp": _now(),
    }


def map_values(df: pd.DataFrame, col: str, mapping_dict: dict):
    new_df = df.copy()
    new_df[col] = new_df[col].replace(mapping_dict)
    return new_df, {
        "operation": "map_values",
        "column": col,
        "params": {"mapping_dict": mapping_dict},
        "timestamp": _now(),
    }


def group_rare_categories(
    df: pd.DataFrame, col: str, threshold: float = 0.01, replacement: str = "Other"
):
    new_df = df.copy()
    freq = new_df[col].value_counts(normalize=True)
    rare = freq[freq < threshold].index
    new_df[col] = new_df[col].where(~new_df[col].isin(rare), other=replacement)
    return new_df, {
        "operation": "group_rare_categories",
        "column": col,
        "params": {
            "threshold": threshold,
            "replacement": replacement,
            "rare_values": rare.tolist(),
        },
        "timestamp": _now(),
    }


def cap_outliers_iqr(df: pd.DataFrame, col: str):
    new_df = df.copy()
    q1 = new_df[col].quantile(0.25)
    q3 = new_df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    new_df[col] = new_df[col].clip(lower=lower, upper=upper)
    return new_df, {
        "operation": "cap_outliers_iqr",
        "column": col,
        "params": {"lower_bound": lower, "upper_bound": upper},
        "timestamp": _now(),
    }


def remove_outlier_rows(
    df: pd.DataFrame, col: str, method: str = "iqr", zscore_threshold: float = 3
):
    new_df = df.copy()
    if method == "iqr":
        q1 = new_df[col].quantile(0.25)
        q3 = new_df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (new_df[col] >= q1 - 1.5 * iqr) & (new_df[col] <= q3 + 1.5 * iqr)
    elif method == "zscore":
        z = (new_df[col] - new_df[col].mean()) / new_df[col].std(ddof=0)
        mask = z.abs() <= zscore_threshold
    else:
        raise ValueError(f"Unknown method: {method}")
    result = new_df[mask].reset_index(drop=True)
    dropped = len(new_df) - len(result)
    return result, {
        "operation": "remove_outlier_rows",
        "column": col,
        "params": {"method": method, "zscore_threshold": zscore_threshold, "rows_dropped": dropped},
        "timestamp": _now(),
    }


def normalize_minmax(df: pd.DataFrame, col: str):
    new_df = df.copy()
    col_min = new_df[col].min()
    col_max = new_df[col].max()
    denom = col_max - col_min
    new_df[col] = (new_df[col] - col_min) / denom if (not pd.isna(denom) and denom != 0) else 0.0
    return new_df, {
        "operation": "normalize_minmax",
        "column": col,
        "params": {"min": col_min, "max": col_max},
        "timestamp": _now(),
    }


def normalize_zscore(df: pd.DataFrame, col: str):
    new_df = df.copy()
    mean = new_df[col].mean()
    std = new_df[col].std(ddof=0)
    new_df[col] = (new_df[col] - mean) / std if (not pd.isna(std) and std != 0) else 0.0
    return new_df, {
        "operation": "normalize_zscore",
        "column": col,
        "params": {"mean": mean, "std": std},
        "timestamp": _now(),
    }


def rename_column(df: pd.DataFrame, old_name: str, new_name: str):
    new_df = df.rename(columns={old_name: new_name})
    return new_df, {
        "operation": "rename_column",
        "column": old_name,
        "params": {"old_name": old_name, "new_name": new_name},
        "timestamp": _now(),
    }


def drop_columns(df: pd.DataFrame, cols: list):
    new_df = df.drop(columns=cols)
    return new_df, {
        "operation": "drop_columns",
        "column": cols,
        "params": {"cols": cols},
        "timestamp": _now(),
    }


def create_column(df: pd.DataFrame, new_col: str, formula: str):
    new_df = df.copy()
    new_df[new_col] = new_df.eval(formula)
    return new_df, {
        "operation": "create_column",
        "column": new_col,
        "params": {"formula": formula},
        "timestamp": _now(),
    }


def bin_column(
    df: pd.DataFrame,
    col: str,
    bins: int,
    labels=None,
    strategy: str = "equal_width",
):
    new_df = df.copy()
    if strategy == "equal_width":
        new_df[col] = pd.cut(new_df[col], bins=bins, labels=labels)
    elif strategy == "quantile":
        new_df[col] = pd.qcut(new_df[col], q=bins, labels=labels, duplicates="drop")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return new_df, {
        "operation": "bin_column",
        "column": col,
        "params": {"bins": bins, "labels": labels, "strategy": strategy},
        "timestamp": _now(),
    }
