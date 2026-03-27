import pandas as pd


def validate_numeric_range(
    df: pd.DataFrame, col: str, min_val=None, max_val=None
) -> pd.DataFrame:
    """Return rows where col value falls outside [min_val, max_val]."""
    mask = pd.Series([False] * len(df), index=df.index)
    if min_val is not None:
        mask |= df[col] < min_val
    if max_val is not None:
        mask |= df[col] > max_val
    return df[mask].copy()


def validate_category_whitelist(
    df: pd.DataFrame, col: str, allowed_values: list
) -> pd.DataFrame:
    """Return rows where col value is not in allowed_values."""
    mask = ~df[col].isin(allowed_values)
    return df[mask].copy()


def validate_non_null(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return rows where col is null."""
    return df[df[col].isnull()].copy()


def run_all_validations(df: pd.DataFrame, rules: list) -> pd.DataFrame:
    """
    Run multiple validation rules and return combined violations.

    Each rule dict must have: col, type (range/whitelist/non_null), params.
    Returns a DataFrame with an additional 'rule' column describing the violation.
    """
    frames = []
    for rule in rules:
        col = rule["col"]
        rule_type = rule["type"]
        params = rule.get("params", {})

        if rule_type == "range":
            violations = validate_numeric_range(
                df, col, min_val=params.get("min_val"), max_val=params.get("max_val")
            )
        elif rule_type == "whitelist":
            violations = validate_category_whitelist(
                df, col, allowed_values=params.get("allowed_values", [])
            )
        elif rule_type == "non_null":
            violations = validate_non_null(df, col)
        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

        if not violations.empty:
            violations = violations.copy()
            violations["rule"] = f"{rule_type} on '{col}'"
            frames.append(violations)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=list(df.columns) + ["rule"])
