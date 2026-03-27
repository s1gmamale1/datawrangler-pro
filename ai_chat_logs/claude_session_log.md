# AI Chat Log — Claude Code (claude-opus-4-6)
### Tool: Claude Code CLI | Date: 2026-03-27 | Project: DataWrangler Pro

---

## Prompt 1 — Project Architecture

**User:**
> I have a Streamlit coursework project for 5COSC038C Data Wrangling & Visualization. I need to build a data preparation studio app with 4 pages: Upload & Overview, Cleaning Studio, Visualization Builder, and Export & Report. What's the best way to structure this?

**Claude:**
> The cleanest approach is to separate concerns into three layers:
> - `utils/` — pure Python functions for all data logic (profiling, cleaning, validation). No Streamlit here.
> - `pages/` — Streamlit UI files, one per page. Each reads/writes `st.session_state`.
> - `app.py` — entry point, initialises session state, wires pages via `st.navigation`.
>
> The key design decision is session state: store `df` (working copy), `original_df` (safety net), `transform_log` (audit trail), and `df_history` (undo stack). Every cleaning operation should push to `df_history` before modifying `df`.

---

## Prompt 2 — Utility Functions

**User:**
> Write the cleaners.py utility file. I need functions for: drop missing rows/cols, fill missing (mean/median/mode/ffill/bfill/constant), drop duplicates, type conversion (handle dirty strings with $ and commas), standardize categorical (trim/lower/upper/title), map values, group rare categories, cap outliers IQR, remove outlier rows, min-max normalize, z-score normalize, rename column, drop columns, create column via formula, bin column. Each function should return (new_df, log_entry_dict) with a UTC timestamp.

**Claude:**
> Here's the full `cleaners.py`...
> *(16 functions implemented, each returning `(modified_df, {"operation": str, "column": str, "params": dict, "timestamp": ISO8601})`)*
>
> Key implementation notes:
> - `convert_type` strips `[\$,\s]` via regex before `pd.to_numeric(..., errors='coerce')` to handle dirty currency strings
> - `bin_column` supports both `pd.cut` (equal width) and `pd.qcut` (quantile) strategies
> - All timestamps use `datetime.now(timezone.utc).isoformat()` for UTC consistency

---

## Prompt 3 — Data Profiler

**User:**
> Write profiler.py with a profile_dataframe function that returns shape, dtypes, missing counts and percentages per column, duplicate count, memory usage, and numeric summary. Also add get_outliers_iqr and get_outliers_zscore functions.

**Claude:**
> `profile_dataframe` returns a dict with all requested keys. Missing % is rounded to 2 decimal places. `get_outliers_iqr` uses the standard 1.5×IQR rule. `get_outliers_zscore` accepts a configurable threshold (default 3).

---

## Prompt 4 — Page A (Upload & Overview)

**User:**
> Build the upload page. It should accept CSV, Excel, JSON. Show 4 metric tiles (rows, cols, missing %, duplicates). Add tabs for Schema, Statistics, Missing Values, Sample Data. Add a data quality score (0-100) based on missing %, duplicate %, and type issues. Use st.cache_data for file reading.

**Claude:**
> Important: use `uploaded_file.getvalue()` not `uploaded_file.read()` — the latter drains the buffer and returns empty bytes on subsequent reruns. Also gate the session state update on filename change:
> ```python
> if st.session_state.get("filename") != uploaded_file.name:
>     # only reload on new file, preserves cleaning work on reruns
> ```
> Quality score formula: `100 - (missing_pct * 0.5) - (duplicate_pct * 0.3) - (type_issue_pct * 0.2)`

---

## Prompt 5 — Page B (Cleaning Studio)

**User:**
> Build the main cleaning page with 8 expanders: missing values, duplicate detection, type conversion, categorical tools (standardize/map/rare grouping), numeric cleaning with outlier detection, normalization, column operations (rename/drop/create/bin), and data validation rules. Add undo functionality.

**Claude:**
> Undo implementation uses a history stack in session state:
> ```python
> def _commit(new_df, log_entry):
>     st.session_state.df_history.append(st.session_state.df.copy())
>     st.session_state.df = new_df
>     st.session_state.transform_log.append(log_entry)
> ```
> Undo button pops both `df_history` and `transform_log` simultaneously to keep them in sync.

---

## Prompt 6 — Page C (Visualization Builder)

**User:**
> Build the visualization page with 8 chart types using Plotly Express: Histogram, Box Plot, Scatter, Line, Bar, Heatmap, Violin, Pie. Add sidebar filters (up to 3 dynamic column filters), optional groupby aggregation, and a saved charts gallery.

**Claude:**
> Used `plotly.express` throughout for interactivity. Heatmap supports two modes: correlation matrix (`df.corr()` on numeric cols) and manual pivot. Bar chart handles its own aggregation before plotting to avoid double-grouping when color is set. Sidebar filters adapt widget type to column dtype — multiselect for categorical, range slider for numeric, date range for datetime.

---

## Prompt 7 — Page D (Export & Report)

**User:**
> Build the export page. Need: CSV/Excel/JSON download with format options, transformation report table with CSV and JSON download, Python recipe generator that converts the log to runnable pandas code, and a session summary with quality improvement delta.

**Claude:**
> Excel export uses `BytesIO` + `openpyxl`:
> ```python
> buffer = io.BytesIO()
> with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
>     df.to_excel(writer, index=False)
> file_data = buffer.getvalue()
> ```
> Recipe generator maps each operation name to its pandas equivalent. Unknown operations emit a `# WARNING` comment rather than crashing.

---

## Prompt 8 — Bug Fixes

**User:**
> Found bugs: (1) navigating back to Page A wipes all cleaning work, (2) recipe generator producing wrong code for fill_missing and convert_type operations.

**Claude:**
> Bug 1 — Root cause: `st.session_state.transform_log = []` was running on every rerun whenever a file was present, not just on new uploads. Fix: gate on filename change.
>
> Bug 2 — Root cause: recipe generator was reading `params.get("method")` but the cleaner logs the key as `strategy`. Similarly `dtype` should be `target_type` and `columns` should be `cols`. Fixed all three key mismatches.

---

## Prompt 9 — Sample Data Generation

**User:**
> Generate two sample CSV datasets for testing. sales_data.csv (1200 rows, 12 cols) and student_records.csv (1100 rows, 12 cols). Both should have intentional data quality issues: missing values, duplicates, outliers, dirty strings, mixed casing.

**Claude:**
> Generated both datasets with seeded randomness (`np.random.seed(42)`) for reproducibility. Injected: ~8% missing in revenue/discount_pct, 50 exact duplicate rows, 32 outlier rows in units_sold (>500), 161 dirty unit_price values with `$` prefix, mixed case in region/country columns.

---

## Summary

| # | Prompt Topic | Outcome |
|---|---|---|
| 1 | Architecture planning | 3-layer structure: utils / pages / app.py |
| 2 | cleaners.py | 16 cleaning functions |
| 3 | profiler.py | profile_dataframe + outlier detectors |
| 4 | Page A — Upload | File upload + profiling + quality score |
| 5 | Page B — Cleaning | 8 expanders + undo stack |
| 6 | Page C — Visualization | 8 chart types + filters + gallery |
| 7 | Page D — Export | CSV/Excel/JSON + report + recipe |
| 8 | Bug fixes | Session state + recipe param keys |
| 9 | Sample data | 2 datasets with intentional quality issues |

**Total AI interactions:** 9 major prompts + iterative debugging
**All generated code was reviewed, tested, and validated manually.**
