import streamlit as st
import pandas as pd
import numpy as np
from utils.profiler import profile_dataframe

st.title("Upload & Overview")


@st.cache_data
def read_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(pd.io.common.BytesIO(file_bytes))
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(pd.io.common.BytesIO(file_bytes))
    elif ext == "json":
        return pd.read_json(pd.io.common.BytesIO(file_bytes))
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Upload Section ────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a dataset",
    type=["csv", "xlsx", "xls", "json"],
)

if uploaded_file is not None:
    # Only reload if a new file is uploaded (avoid wiping cleaning work on reruns)
    if st.session_state.get("filename") != uploaded_file.name:
        try:
            df = read_file(uploaded_file.getvalue(), uploaded_file.name)
            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.session_state.filename = uploaded_file.name
            st.session_state.transform_log = []
            st.session_state.df_history = []
        except Exception as e:
            st.error(f"Could not read file: {e}")

if st.session_state.get("filename"):
    st.success(
        f"Loaded **{st.session_state.filename}** — "
        f"{st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]:,} columns"
    )

if st.button("Reset Session"):
    st.session_state.df = None
    st.session_state.original_df = None
    st.session_state.filename = None
    st.session_state.transform_log = []
    st.rerun()

# ── Dataset Overview ──────────────────────────────────────────────────────────

df = st.session_state.get("df")

if df is not None:
    st.divider()
    st.subheader("Dataset Overview")

    profile = profile_dataframe(df)
    n_rows, n_cols = profile["shape"]
    n_duplicates = profile["duplicates"]
    total_cells = n_rows * n_cols
    total_missing = sum(v["count"] for v in profile["missing"].values())
    missing_pct_overall = round(total_missing / total_cells * 100, 2) if total_cells else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{n_rows:,}")
    col2.metric("Columns", f"{n_cols:,}")
    col3.metric("Missing Cells %", f"{missing_pct_overall:.2f}%")
    col4.metric("Duplicate Rows", f"{n_duplicates:,}")

    tab_schema, tab_stats, tab_missing, tab_sample = st.tabs(
        ["Schema", "Statistics", "Missing Values", "Sample Data"]
    )

    with tab_schema:
        schema_rows = []
        for col in df.columns:
            m = profile["missing"][col]
            schema_rows.append(
                {
                    "Column": col,
                    "dtype": profile["dtypes"][col],
                    "Non-Null Count": n_rows - m["count"],
                    "Missing %": m["pct"],
                }
            )
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

    with tab_stats:
        desc = df.describe(include="all").T
        st.dataframe(desc.style.format(precision=3, na_rep="—"), use_container_width=True)

    with tab_missing:
        missing_series = pd.Series(
            {col: v["pct"] for col, v in profile["missing"].items()},
            name="Missing %",
        )
        missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
        if missing_series.empty:
            st.info("No missing values found.")
        else:
            st.bar_chart(missing_series)
            st.dataframe(
                missing_series.reset_index().rename(columns={"index": "Column"}),
                use_container_width=True,
                hide_index=True,
            )

    with tab_sample:
        st.caption(f"Showing first 100 of {n_rows:,} rows")
        st.dataframe(df.head(100), use_container_width=True)

    # ── Data Quality Score ────────────────────────────────────────────────────

    st.divider()
    st.subheader("Data Quality Score")

    duplicate_pct = (n_duplicates / n_rows * 100) if n_rows else 0.0

    # Type issue: object columns that look numeric (>50 % of non-null values parse as float)
    type_issue_count = 0
    for col in df.select_dtypes(include="object").columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        try:
            numeric_ratio = pd.to_numeric(non_null, errors="coerce").notna().mean()
            if numeric_ratio > 0.5:
                type_issue_count += 1
        except Exception:
            pass
    type_issue_pct = (type_issue_count / n_cols * 100) if n_cols else 0.0

    score = 100 - (missing_pct_overall * 0.5) - (duplicate_pct * 0.3) - (type_issue_pct * 0.2)
    score = float(np.clip(score, 0, 100))

    if score > 80:
        color = "green"
        label = "Good"
    elif score >= 50:
        color = "orange"
        label = "Fair"
    else:
        color = "red"
        label = "Poor"

    st.progress(int(score) / 100)
    st.markdown(
        f"<h3 style='color:{color};'>Quality Score: {score:.1f} / 100 — {label}</h3>",
        unsafe_allow_html=True,
    )
    with st.expander("Score breakdown"):
        st.markdown(
            f"""
| Factor | Value | Penalty |
|---|---|---|
| Missing cells % | {missing_pct_overall:.2f}% | −{missing_pct_overall * 0.5:.2f} |
| Duplicate rows % | {duplicate_pct:.2f}% | −{duplicate_pct * 0.3:.2f} |
| Suspected type issues % | {type_issue_pct:.2f}% | −{type_issue_pct * 0.2:.2f} |
| **Final score** | | **{score:.1f}** |
"""
        )
