import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime

st.title("Export & Report")

if st.session_state.get("df") is None:
    st.warning("No data loaded. Please upload a dataset on the Upload page.")
    st.stop()

df = st.session_state.df
original_df = st.session_state.get("original_df", df)
transform_log = st.session_state.get("transform_log", [])
filename = st.session_state.get("filename", "data.csv")
base_name = filename.rsplit(".", 1)[0] if "." in filename else filename


# ── 1. Export Cleaned Dataset ────────────────────────────────────────────────
st.header("1. Export Cleaned Dataset")

orig_rows, orig_cols = original_df.shape
curr_rows, curr_cols = df.shape
rows_removed = orig_rows - curr_rows
cols_removed = orig_cols - curr_cols

col1, col2, col3, col4 = st.columns(4)
col1.metric("Original Shape", f"{orig_rows} × {orig_cols}")
col2.metric("Current Shape", f"{curr_rows} × {curr_cols}")
col3.metric("Rows Removed", rows_removed, delta=-rows_removed if rows_removed else None, delta_color="inverse")
col4.metric("Cols Removed", cols_removed, delta=-cols_removed if cols_removed else None, delta_color="inverse")

st.divider()

export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
export_filename_input = st.text_input("Filename", value=f"cleaned_{base_name}")

if export_format == "CSV":
    delimiter_option = st.radio("Delimiter", [",", ";", "Tab"], horizontal=True)
    delimiter = "\t" if delimiter_option == "Tab" else delimiter_option
    export_ext = ".csv"
    file_data = df.to_csv(index=False, sep=delimiter).encode("utf-8")
    mime = "text/csv"

elif export_format == "Excel":
    export_ext = ".xlsx"
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned Data")
    file_data = buffer.getvalue()
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

else:
    export_ext = ".json"
    file_data = df.to_json(orient="records", indent=2).encode("utf-8")
    mime = "application/json"

final_export_name = export_filename_input if export_filename_input.endswith(export_ext) else export_filename_input + export_ext

st.download_button(
    label=f"Download {export_format}",
    data=file_data,
    file_name=final_export_name,
    mime=mime,
    use_container_width=True,
)


# ── 2. Transformation Report ─────────────────────────────────────────────────
st.header("2. Transformation Report")

if not transform_log:
    st.info("No transformations have been applied yet.")
else:
    report_rows = []
    for i, entry in enumerate(transform_log, start=1):
        report_rows.append({
            "#": i,
            "Timestamp": entry.get("timestamp", ""),
            "Operation": entry.get("operation", ""),
            "Column": entry.get("column", ""),
            "Parameters": json.dumps(entry.get("params", {}), default=str),
        })
    report_df = pd.DataFrame(report_rows).astype(str)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    report_csv = report_df.to_csv(index=False).encode("utf-8")
    report_json = json.dumps(transform_log, indent=2, default=str).encode("utf-8")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="Download Report as CSV",
            data=report_csv,
            file_name=f"{base_name}_transform_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            label="Download Report as JSON",
            data=report_json,
            file_name=f"{base_name}_transform_report.json",
            mime="application/json",
            use_container_width=True,
        )


# ── 3. Recipe Export ─────────────────────────────────────────────────────────
st.header("3. Recipe Export")

def generate_recipe(log, source_filename):
    lines = [
        "import pandas as pd",
        "",
        f'df = pd.read_csv("{source_filename}")',
        "",
    ]

    op_map = {
        "drop_missing_rows": lambda e: _code_drop_missing_rows(e),
        "fill_missing": lambda e: _code_fill_missing(e),
        "drop_duplicates": lambda e: _code_drop_duplicates(e),
        "convert_type": lambda e: _code_convert_type(e),
        "normalize_minmax": lambda e: _code_normalize_minmax(e),
        "normalize_zscore": lambda e: _code_normalize_zscore(e),
        "rename_column": lambda e: _code_rename_column(e),
        "drop_columns": lambda e: _code_drop_columns(e),
    }

    for i, entry in enumerate(log, start=1):
        op = entry.get("operation", "")
        col = entry.get("column", "")
        params = entry.get("params", {})
        ts = entry.get("timestamp", "")
        lines.append(f"# Step {i}: {op} | column: {col} | {ts}")
        handler = op_map.get(op)
        if handler:
            lines.append(handler(entry))
        else:
            lines.append(f"# WARNING: unknown operation '{op}' — skipped")
        lines.append("")

    lines += [
        'df.to_csv("output_cleaned.csv", index=False)',
        'print("Done. Saved to output_cleaned.csv")',
    ]
    return "\n".join(lines)


def _code_drop_missing_rows(entry):
    params = entry.get("params", {})
    col = entry.get("column", "")
    subset = f'subset=["{col}"]' if col else ""
    thresh = params.get("thresh")
    if thresh:
        return f"df = df.dropna(thresh={thresh})"
    if subset:
        return f"df = df.dropna({subset})"
    return "df = df.dropna()"


def _code_fill_missing(entry):
    col = entry.get("column", "")
    params = entry.get("params", {})
    strategy = params.get("strategy", "mean")
    fill_value = params.get("fill_value")
    c = f'"{col}"'
    if strategy == "mean":
        return f"df[{c}] = df[{c}].fillna(df[{c}].mean())"
    elif strategy == "median":
        return f"df[{c}] = df[{c}].fillna(df[{c}].median())"
    elif strategy == "mode":
        return f"df[{c}] = df[{c}].fillna(df[{c}].mode()[0])"
    elif strategy == "constant" and fill_value is not None:
        v = f'"{fill_value}"' if isinstance(fill_value, str) else fill_value
        return f"df[{c}] = df[{c}].fillna({v})"
    elif strategy == "ffill":
        return f"df[{c}] = df[{c}].ffill()"
    elif strategy == "bfill":
        return f"df[{c}] = df[{c}].bfill()"
    else:
        return f"df[{c}] = df[{c}].fillna(df[{c}].mean())"


def _code_drop_duplicates(entry):
    params = entry.get("params", {})
    keep = params.get("keep", "first")
    subset = params.get("subset")
    if subset:
        cols_repr = repr(subset) if isinstance(subset, list) else f'["{subset}"]'
        return f"df = df.drop_duplicates(subset={cols_repr}, keep='{keep}')"
    return f"df = df.drop_duplicates(keep='{keep}')"


def _code_convert_type(entry):
    col = entry.get("column", "")
    params = entry.get("params", {})
    target_type = params.get("target_type", "str")
    c = f'"{col}"'
    return f"df[{c}] = df[{c}].astype({repr(target_type)})"


def _code_normalize_minmax(entry):
    col = entry.get("column", "")
    c = f'"{col}"'
    return f"df[{c}] = (df[{c}] - df[{c}].min()) / (df[{c}].max() - df[{c}].min())"


def _code_normalize_zscore(entry):
    col = entry.get("column", "")
    c = f'"{col}"'
    return f"df[{c}] = (df[{c}] - df[{c}].mean()) / df[{c}].std()"


def _code_rename_column(entry):
    params = entry.get("params", {})
    old = params.get("old_name", entry.get("column", ""))
    new = params.get("new_name", "")
    return f'df = df.rename(columns={{"{old}": "{new}"}})'


def _code_drop_columns(entry):
    params = entry.get("params", {})
    cols = params.get("cols", entry.get("column", []))
    if isinstance(cols, str):
        cols = [cols]
    return f"df = df.drop(columns={repr(cols)})"


recipe_script = generate_recipe(transform_log, filename)
st.code(recipe_script, language="python")

st.download_button(
    label="Download Recipe as .py",
    data=recipe_script.encode("utf-8"),
    file_name=f"{base_name}_recipe.py",
    mime="text/x-python",
    use_container_width=True,
)


# ── 4. Session Summary ───────────────────────────────────────────────────────
st.header("4. Session Summary")

total_ops = len(transform_log)
cols_modified = len({str(e.get("column", "")) for e in transform_log if e.get("column", "")})

null_before = int(original_df.isnull().sum().sum())
null_after = int(df.isnull().sum().sum())
total_cells_before = orig_rows * orig_cols if orig_rows * orig_cols > 0 else 1
null_before_pct = null_before / total_cells_before * 100
null_after_pct = null_after / (curr_rows * curr_cols) * 100 if curr_rows * curr_cols > 0 else 0
quality_score = round(min(100.0, max(0.0, 100.0 - null_after_pct)), 1)

s1, s2, s3, s4 = st.columns(4)
s1.metric("Total Operations Applied", total_ops)
s2.metric("Columns Modified", cols_modified)
s3.metric("Rows Removed", rows_removed)
s4.metric("Data Quality Score", f"{quality_score}%", delta=f"{round(null_before_pct - null_after_pct, 1)}pp null reduction")

st.divider()

if st.button("Start New Session", type="primary", use_container_width=True):
    for key in ["df", "original_df", "transform_log", "filename"]:
        if key in st.session_state:
            del st.session_state[key]
    st.success("Session cleared. Return to the Upload page to load a new dataset.")
    st.rerun()
