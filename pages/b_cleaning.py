import pandas as pd
import streamlit as st

from utils.cleaners import (
    bin_column,
    cap_outliers_iqr,
    convert_type,
    create_column,
    drop_columns,
    drop_duplicates,
    drop_missing_cols,
    drop_missing_rows,
    fill_missing,
    group_rare_categories,
    map_values,
    normalize_minmax,
    normalize_zscore,
    remove_outlier_rows,
    rename_column,
    standardize_categorical,
)
from utils.profiler import get_outliers_iqr, get_outliers_zscore, profile_dataframe
from utils.validators import run_all_validations

st.title("Data Cleaning")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Upload a dataset on Page A first")
    st.stop()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "df_history" not in st.session_state:
    st.session_state.df_history = []
if "transform_log" not in st.session_state:
    st.session_state.transform_log = []
if "validation_rules" not in st.session_state:
    st.session_state.validation_rules = []


def _commit(new_df: pd.DataFrame, log_entry: dict):
    st.session_state.df_history.append(st.session_state.df.copy())
    st.session_state.df = new_df
    st.session_state.transform_log.append(log_entry)


# ---------------------------------------------------------------------------
# Shape + Undo
# ---------------------------------------------------------------------------
df = st.session_state.df
col_shape, col_undo = st.columns([3, 1])
col_shape.metric("Current shape", f"{df.shape[0]} rows × {df.shape[1]} cols")

if col_undo.button("↩ Undo Last Step", disabled=len(st.session_state.df_history) == 0):
    st.session_state.df = st.session_state.df_history.pop()
    if st.session_state.transform_log:
        st.session_state.transform_log.pop()
    st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# 1. Missing Values
# ---------------------------------------------------------------------------
with st.expander("1. Missing Values", expanded=False):
    df = st.session_state.df
    profile = profile_dataframe(df)
    missing_data = [
        {"Column": c, "Missing Count": v["count"], "Missing %": v["pct"]}
        for c, v in profile["missing"].items()
        if v["count"] > 0
    ]
    if missing_data:
        st.dataframe(pd.DataFrame(missing_data), use_container_width=True)
    else:
        st.success("No missing values found.")

    action = st.radio(
        "Action",
        ["Drop rows", "Drop columns (by threshold)", "Fill values"],
        horizontal=True,
        key="mv_action",
    )

    if action == "Drop rows":
        subset_cols = st.multiselect(
            "Subset columns (blank = all)", df.columns.tolist(), key="mv_drop_rows_subset"
        )
        if st.button("Apply – Drop Rows", key="mv_apply_drop_rows"):
            subset = subset_cols if subset_cols else None
            new_df, entry = drop_missing_rows(df, subset=subset)
            _commit(new_df, entry)
            st.success(
                f"Dropped {entry['params']['rows_dropped']} rows. "
                f"Before: {len(df)} → After: {len(new_df)}"
            )
            st.rerun()

    elif action == "Drop columns (by threshold)":
        threshold = st.slider(
            "Drop columns where missing % >", 0.0, 1.0, 0.5, 0.05, key="mv_thresh"
        )
        if st.button("Apply – Drop Columns", key="mv_apply_drop_cols"):
            new_df, entry = drop_missing_cols(df, threshold=threshold)
            _commit(new_df, entry)
            st.toast(f"Dropped columns: {entry['params']['cols_dropped']}", icon="✅")
            st.rerun()

    else:
        fill_cols = st.multiselect(
            "Select column(s) to fill", df.columns.tolist(), key="mv_fill_cols"
        )
        strategy = st.selectbox(
            "Strategy",
            ["constant", "mean", "median", "mode", "ffill", "bfill"],
            key="mv_strategy",
        )
        fill_value = None
        if strategy == "constant":
            fill_value = st.text_input("Fill value", key="mv_fill_value")

        if st.button("Apply – Fill", key="mv_apply_fill") and fill_cols:
            current_df = st.session_state.df
            for col_name in fill_cols:
                current_df, entry = fill_missing(
                    current_df, col_name, strategy, fill_value=fill_value
                )
                st.session_state.df_history.append(st.session_state.df.copy())
                st.session_state.transform_log.append(entry)
            st.session_state.df = current_df
            st.toast(f"Filled missing values in: {fill_cols}", icon="✅")
            st.rerun()

# ---------------------------------------------------------------------------
# 2. Duplicate Detection
# ---------------------------------------------------------------------------
with st.expander("2. Duplicate Detection", expanded=False):
    df = st.session_state.df
    total_dups = int(df.duplicated().sum())
    st.metric("Total duplicate rows", total_dups)

    dup_subset = st.multiselect(
        "Subset columns (blank = all columns)", df.columns.tolist(), key="dup_subset"
    )
    keep_opt = st.radio("Keep", ["first", "last"], horizontal=True, key="dup_keep")

    subset_for_check = dup_subset if dup_subset else None
    dup_rows = df[df.duplicated(subset=subset_for_check, keep=False)]
    with st.expander(f"Preview duplicate rows ({len(dup_rows)})"):
        st.dataframe(dup_rows, use_container_width=True)

    if st.button("Remove Duplicates", key="dup_apply"):
        new_df, entry = drop_duplicates(df, subset=subset_for_check, keep=keep_opt)
        _commit(new_df, entry)
        st.toast(f"Removed {entry['params']['rows_dropped']} duplicate rows.", icon="✅")
        st.rerun()

# ---------------------------------------------------------------------------
# 3. Type Conversion
# ---------------------------------------------------------------------------
with st.expander("3. Type Conversion", expanded=False):
    df = st.session_state.df
    tc_col = st.selectbox("Select column", df.columns.tolist(), key="tc_col")
    st.caption(f"Current dtype: **{df[tc_col].dtype}**")

    target_type = st.selectbox(
        "Target type",
        ["integer", "float", "string", "category", "datetime"],
        key="tc_target",
    )

    dt_format = None
    if target_type == "datetime":
        dt_format = st.text_input("Datetime format (e.g. %Y-%m-%d)", key="tc_dtfmt")
        if not dt_format:
            dt_format = None

    cleaner_type_map = {
        "integer": "numeric",
        "float": "numeric",
        "string": "string",
        "category": "category",
        "datetime": "datetime",
    }

    if st.button("Convert", key="tc_apply"):
        try:
            mapped = cleaner_type_map[target_type]
            new_df, entry = convert_type(df, tc_col, mapped, datetime_format=dt_format)
            if target_type == "integer" and mapped == "numeric":
                new_df[tc_col] = pd.to_numeric(new_df[tc_col], errors="coerce").astype(
                    "Int64"
                )
            _commit(new_df, entry)
            st.toast(f"Converted '{tc_col}' to {target_type}.", icon="✅")
            st.rerun()
        except Exception as e:
            st.error(f"Conversion failed: {e}")

# ---------------------------------------------------------------------------
# 4. Categorical Tools
# ---------------------------------------------------------------------------
with st.expander("4. Categorical Tools", expanded=False):
    df = st.session_state.df
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        st.info("No categorical / object columns found.")
    else:
        cat_col = st.selectbox("Select column", cat_cols, key="cat_col")
        st.write("**Value counts**")
        vc = df[cat_col].value_counts(dropna=False).reset_index()
        vc.columns = ["Value", "Count"]
        st.dataframe(vc, use_container_width=True)

        tab_std, tab_map, tab_rare = st.tabs(["Standardize", "Map Values", "Group Rare"])

        with tab_std:
            trim = st.checkbox("Trim whitespace", key="std_trim")
            lower = st.checkbox("Lowercase", key="std_lower")
            upper = st.checkbox("Uppercase", key="std_upper")
            title = st.checkbox("Title case", key="std_title")
            if st.button("Apply Standardize", key="std_apply"):
                ops = []
                if trim:
                    ops.append("trim")
                if lower:
                    ops.append("lower")
                if upper:
                    ops.append("upper")
                if title:
                    ops.append("title")
                if ops:
                    new_df, entry = standardize_categorical(df, cat_col, ops)
                    _commit(new_df, entry)
                    st.toast(f"Applied: {ops}", icon="✅")
                    st.rerun()
                else:
                    st.warning("Select at least one operation.")

        with tab_map:
            unique_vals = df[cat_col].dropna().unique().tolist()
            map_df = pd.DataFrame({"Value": unique_vals, "Replacement": unique_vals})
            edited = st.data_editor(map_df, use_container_width=True, key="cat_map_editor")
            if st.button("Apply Map Values", key="map_apply"):
                mapping_dict = dict(zip(edited["Value"], edited["Replacement"]))
                mapping_dict = {k: v for k, v in mapping_dict.items() if str(k) != str(v)}
                if mapping_dict:
                    new_df, entry = map_values(df, cat_col, mapping_dict)
                    _commit(new_df, entry)
                    st.toast(f"Mapped {len(mapping_dict)} value(s).", icon="✅")
                    st.rerun()
                else:
                    st.info("No changes detected in the mapping table.")

        with tab_rare:
            rare_thresh = st.slider(
                "Frequency threshold", 0.01, 0.20, 0.05, 0.01, key="rare_thresh"
            )
            rare_replacement = st.text_input(
                "Replacement label", value="Other", key="rare_replace"
            )
            freq = df[cat_col].value_counts(normalize=True)
            rare_preview = freq[freq < rare_thresh].index.tolist()
            st.caption(f"Categories to be grouped ({len(rare_preview)}): {rare_preview}")
            if st.button("Apply Group Rare", key="rare_apply"):
                new_df, entry = group_rare_categories(
                    df, cat_col, threshold=rare_thresh, replacement=rare_replacement
                )
                _commit(new_df, entry)
                st.toast(f"Grouped {len(rare_preview)} rare categories → '{rare_replacement}'.", icon="✅")
                st.rerun()

# ---------------------------------------------------------------------------
# 5. Numeric Cleaning (Outliers)
# ---------------------------------------------------------------------------
with st.expander("5. Numeric Cleaning", expanded=False):
    df = st.session_state.df
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("No numeric columns found.")
    else:
        num_col = st.selectbox("Select numeric column", num_cols, key="nc_col")
        series = df[num_col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        stats_df = pd.DataFrame(
            {
                "Stat": ["Min", "Max", "Mean", "IQR Lower", "IQR Upper"],
                "Value": [
                    series.min(),
                    series.max(),
                    series.mean(),
                    q1 - 1.5 * iqr,
                    q3 + 1.5 * iqr,
                ],
            }
        )
        st.dataframe(stats_df, use_container_width=True)

        method = st.radio("Outlier method", ["IQR", "Z-score"], horizontal=True, key="nc_method")
        zscore_thresh = 3.0
        if method == "Z-score":
            zscore_thresh = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1, key="nc_zthresh")

        if method == "IQR":
            outlier_mask = get_outliers_iqr(df, num_col)
        else:
            outlier_mask = get_outliers_zscore(df, num_col, threshold=zscore_thresh)

        st.metric("Outlier count", int(outlier_mask.sum()))

        nc_action = st.radio(
            "Action", ["Cap (winsorize)", "Remove rows", "Ignore"],
            horizontal=True,
            key="nc_action",
        )

        if st.button("Apply Numeric Cleaning", key="nc_apply"):
            if nc_action == "Ignore":
                st.info("No action taken.")
            elif nc_action == "Cap (winsorize)":
                new_df, entry = cap_outliers_iqr(df, num_col)
                _commit(new_df, entry)
                st.toast(f"Capped outliers in '{num_col}'.", icon="✅")
                st.rerun()
            else:
                m = "iqr" if method == "IQR" else "zscore"
                new_df, entry = remove_outlier_rows(
                    df, num_col, method=m, zscore_threshold=zscore_thresh
                )
                _commit(new_df, entry)
                st.toast(f"Removed {entry['params']['rows_dropped']} outlier rows.", icon="✅")
                st.rerun()

# ---------------------------------------------------------------------------
# 6. Normalization / Scaling
# ---------------------------------------------------------------------------
with st.expander("6. Normalization / Scaling", expanded=False):
    df = st.session_state.df
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.info("No numeric columns found.")
    else:
        scale_cols = st.multiselect("Select column(s)", num_cols, key="scale_cols")
        scale_method = st.radio(
            "Method", ["Min-Max", "Z-score"], horizontal=True, key="scale_method"
        )

        if scale_cols:
            before_stats = df[scale_cols].agg(["min", "max", "mean", "std"]).T
            before_stats.columns = ["Before Min", "Before Max", "Before Mean", "Before Std"]
            st.write("**Before stats**")
            st.dataframe(before_stats.round(4), use_container_width=True)

        if st.button("Apply Normalization", key="scale_apply") and scale_cols:
            current_df = st.session_state.df
            for c in scale_cols:
                if scale_method == "Min-Max":
                    current_df, entry = normalize_minmax(current_df, c)
                else:
                    current_df, entry = normalize_zscore(current_df, c)
                st.session_state.df_history.append(st.session_state.df.copy())
                st.session_state.transform_log.append(entry)
            st.session_state.df = current_df

            after_stats = current_df[scale_cols].agg(["min", "max", "mean", "std"]).T
            after_stats.columns = ["After Min", "After Max", "After Mean", "After Std"]
            st.write("**After stats**")
            st.dataframe(after_stats.round(4), use_container_width=True)
            st.toast(f"Normalized {scale_cols} using {scale_method}.", icon="✅")
            st.rerun()

# ---------------------------------------------------------------------------
# 7. Column Operations
# ---------------------------------------------------------------------------
with st.expander("7. Column Operations", expanded=False):
    df = st.session_state.df

    col_rename, col_drop, col_new = st.columns(3)

    with col_rename:
        st.subheader("Rename")
        old_col = st.selectbox("Column to rename", df.columns.tolist(), key="ren_old")
        new_col_name = st.text_input("New name", key="ren_new")
        if st.button("Rename", key="ren_apply"):
            if new_col_name and new_col_name != old_col:
                if new_col_name in df.columns:
                    st.error("Column name already exists.")
                else:
                    new_df, entry = rename_column(df, old_col, new_col_name)
                    _commit(new_df, entry)
                    st.toast(f"Renamed '{old_col}' → '{new_col_name}'.", icon="✅")
                    st.rerun()
            else:
                st.warning("Enter a different name.")

    with col_drop:
        st.subheader("Drop")
        drop_cols_sel = st.multiselect("Columns to drop", df.columns.tolist(), key="drop_cols_sel")
        confirm_drop = st.checkbox("Confirm drop", key="drop_confirm")
        if st.button("Drop Columns", key="drop_apply"):
            if drop_cols_sel and confirm_drop:
                new_df, entry = drop_columns(df, drop_cols_sel)
                _commit(new_df, entry)
                st.toast(f"Dropped: {drop_cols_sel}", icon="✅")
                st.rerun()
            elif not drop_cols_sel:
                st.warning("Select columns to drop.")
            else:
                st.warning("Check the confirm box first.")

    with col_new:
        st.subheader("New Column")
        new_col_label = st.text_input("Column name", key="new_col_name")
        formula = st.text_input(
            "Formula (e.g. col_a * 2 + col_b)", key="new_col_formula"
        )
        if st.button("Create", key="new_col_apply"):
            if new_col_label and formula:
                try:
                    new_df, entry = create_column(df, new_col_label, formula)
                    _commit(new_df, entry)
                    st.toast(f"Created column '{new_col_label}'.", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Formula error: {e}")
            else:
                st.warning("Provide both a column name and a formula.")

    st.divider()
    st.subheader("Binning")
    bin_num_cols = df.select_dtypes(include="number").columns.tolist()
    if not bin_num_cols:
        st.info("No numeric columns available for binning.")
    else:
        bin_col_sel = st.selectbox("Numeric column", bin_num_cols, key="bin_col")
        n_bins = st.slider("Number of bins", 2, 20, 5, key="bin_n")
        bin_strategy = st.radio(
            "Strategy", ["equal_width", "quantile"], horizontal=True, key="bin_strat"
        )
        bin_labels_raw = st.text_input(
            "Labels (comma-separated, optional)", key="bin_labels"
        )
        bin_labels = (
            [l.strip() for l in bin_labels_raw.split(",") if l.strip()]
            if bin_labels_raw
            else None
        )
        if bin_labels and len(bin_labels) != n_bins:
            st.warning(f"Number of labels ({len(bin_labels)}) must match bins ({n_bins}).")
            bin_labels = None

        if st.button("Apply Binning", key="bin_apply"):
            try:
                new_df, entry = bin_column(
                    df, bin_col_sel, bins=n_bins, labels=bin_labels, strategy=bin_strategy
                )
                _commit(new_df, entry)
                st.toast(f"Binned '{bin_col_sel}' into {n_bins} bins.", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Binning error: {e}")

# ---------------------------------------------------------------------------
# 8. Data Validation Rules
# ---------------------------------------------------------------------------
with st.expander("8. Data Validation Rules", expanded=False):
    df = st.session_state.df

    if st.button("+ Add Rule", key="val_add"):
        st.session_state.validation_rules.append(
            {"col": df.columns[0], "type": "non_null", "params": {}}
        )

    rules = st.session_state.validation_rules
    updated_rules = []

    for i, rule in enumerate(rules):
        st.markdown(f"**Rule {i + 1}**")
        rcol1, rcol2, rcol3, rcol4 = st.columns([2, 2, 2, 1])

        with rcol1:
            sel_col = st.selectbox(
                "Column", df.columns.tolist(), index=df.columns.tolist().index(rule["col"])
                if rule["col"] in df.columns else 0,
                key=f"val_col_{i}",
            )
        with rcol2:
            rule_type = st.selectbox(
                "Rule type",
                ["non_null", "range", "whitelist"],
                index=["non_null", "range", "whitelist"].index(rule["type"])
                if rule["type"] in ["non_null", "range", "whitelist"] else 0,
                key=f"val_type_{i}",
            )
        params = {}
        with rcol3:
            if rule_type == "range":
                min_v = st.text_input("Min (optional)", key=f"val_min_{i}")
                max_v = st.text_input("Max (optional)", key=f"val_max_{i}")
                params["min_val"] = float(min_v) if min_v else None
                params["max_val"] = float(max_v) if max_v else None
            elif rule_type == "whitelist":
                allowed_raw = st.text_input(
                    "Allowed values (comma-separated)", key=f"val_wl_{i}"
                )
                params["allowed_values"] = (
                    [v.strip() for v in allowed_raw.split(",") if v.strip()]
                    if allowed_raw
                    else []
                )
        with rcol4:
            remove_rule = st.button("Remove", key=f"val_rm_{i}")

        if not remove_rule:
            updated_rules.append({"col": sel_col, "type": rule_type, "params": params})

    st.session_state.validation_rules = updated_rules

    if st.button("Run Validation", key="val_run"):
        if not st.session_state.validation_rules:
            st.info("Add at least one rule before running validation.")
        else:
            try:
                violations = run_all_validations(df, st.session_state.validation_rules)
                if violations.empty:
                    st.success("No violations found.")
                else:
                    st.warning(f"{len(violations)} violation(s) found.")
                    st.dataframe(violations, use_container_width=True)
                    csv_bytes = violations.to_csv(index=False).encode()
                    st.download_button(
                        "Download violations as CSV",
                        data=csv_bytes,
                        file_name="violations.csv",
                        mime="text/csv",
                        key="val_download",
                    )
            except Exception as e:
                st.error(f"Validation error: {e}")

# ---------------------------------------------------------------------------
# Transform Log
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Transform Log", expanded=False):
    log = st.session_state.transform_log
    if log:
        log_df = pd.DataFrame(
            [
                {
                    "Step": i + 1,
                    "Operation": e.get("operation"),
                    "Column": str(e.get("column")),
                    "Params": str(e.get("params")),
                    "Timestamp": e.get("timestamp"),
                }
                for i, e in enumerate(log)
            ]
        )
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("No transformations applied yet.")
