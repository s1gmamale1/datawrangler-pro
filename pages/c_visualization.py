import streamlit as st
import pandas as pd
import plotly.express as px

st.title("C — Visualization")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("No data loaded. Please upload a dataset on the Upload page.")
    st.stop()

if "saved_charts" not in st.session_state:
    st.session_state.saved_charts = []

df_orig = st.session_state.df.copy()

# ── helpers ──────────────────────────────────────────────────────────────────
def col_type(df, col):
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return "datetime"
    if pd.api.types.is_numeric_dtype(df[col]):
        return "numeric"
    return "categorical"

def apply_filters(df, filters):
    for f in filters:
        col = f.get("col")
        kind = f.get("kind")
        val = f.get("val")
        if col is None or col not in df.columns or val is None:
            continue
        if kind == "categorical":
            if val:
                df = df[df[col].isin(val)]
        elif kind == "numeric":
            df = df[df[col].between(val[0], val[1])]
        elif kind == "datetime":
            df = df[(df[col] >= pd.Timestamp(val[0])) & (df[col] <= pd.Timestamp(val[1]))]
    return df

def build_chart(df, cfg):
    ctype = cfg["chart_type"]
    try:
        if ctype == "Histogram":
            kw = dict(x=cfg["x"], nbins=cfg["bins"])
            if cfg.get("color"):
                kw["color"] = cfg["color"]
            return px.histogram(df, **kw)

        elif ctype == "Box Plot":
            kw = dict(y=cfg["y"])
            if cfg.get("x"):
                kw["x"] = cfg["x"]
            if cfg.get("color"):
                kw["color"] = cfg["color"]
            return px.box(df, **kw)

        elif ctype == "Scatter Plot":
            kw = dict(x=cfg["x"], y=cfg["y"])
            if cfg.get("color"):
                kw["color"] = cfg["color"]
            if cfg.get("size"):
                kw["size"] = cfg["size"]
            if cfg.get("trendline"):
                kw["trendline"] = "ols"
            return px.scatter(df, **kw)

        elif ctype == "Line Chart":
            kw = dict(x=cfg["x"], y=cfg["y_cols"])
            if cfg.get("color"):
                kw["color"] = cfg["color"]
            return px.line(df, **kw)

        elif ctype == "Bar Chart":
            agg_fn = cfg.get("bar_agg", "sum")
            grp_cols = [c for c in [cfg["x"], cfg.get("color")] if c]
            if cfg["y"] in grp_cols:
                grp_cols = [cfg["x"]]
            agg_map = {"sum": "sum", "mean": "mean", "count": "count"}
            dfa = df.groupby(grp_cols, dropna=False)[cfg["y"]].agg(agg_map[agg_fn]).reset_index()
            kw = dict(x=cfg["x"], y=cfg["y"])
            if cfg.get("color") and cfg["color"] in dfa.columns:
                kw["color"] = cfg["color"]
            if cfg.get("orientation") == "h":
                kw["x"], kw["y"] = kw["y"], kw["x"]
                kw["orientation"] = "h"
            return px.bar(dfa, **kw)

        elif ctype == "Heatmap":
            if cfg.get("corr_matrix"):
                num_df = df.select_dtypes("number")
                corr = num_df.corr()
                return px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            else:
                pivot = df.pivot_table(index=cfg["y"], columns=cfg["x"], values=cfg["z"], aggfunc="mean")
                return px.imshow(pivot, text_auto=True)

        elif ctype == "Violin Plot":
            kw = dict(y=cfg["y"], x=cfg["x"])
            if cfg.get("color"):
                kw["color"] = cfg["color"]
            return px.violin(df, **kw, box=True, points="outliers")

        elif ctype == "Pie Chart":
            top_n = cfg.get("top_n", 10)
            dfa = df.groupby(cfg["names"])[cfg["values"]].sum().nlargest(top_n).reset_index()
            return px.pie(dfa, values=cfg["values"], names=cfg["names"])

    except Exception as e:
        st.error(f"Chart error: {e}")
        return None

# ── sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Column Filters")

filters = []
all_cols = list(df_orig.columns)

for i in range(3):
    with st.sidebar.expander(f"Filter {i + 1}", expanded=(i == 0)):
        col_choice = st.selectbox(f"Column ##{i}", ["(none)"] + all_cols, key=f"fcol_{i}")
        if col_choice != "(none)":
            kind = col_type(df_orig, col_choice)
            if kind == "categorical":
                options = sorted(df_orig[col_choice].dropna().unique().tolist(), key=str)
                val = st.multiselect(f"Values ##{i}", options, default=options, key=f"fval_{i}")
            elif kind == "numeric":
                mn = float(df_orig[col_choice].min())
                mx = float(df_orig[col_choice].max())
                if mn == mx:
                    st.info("Column has a single unique value.")
                    val = (mn, mx)
                else:
                    val = st.slider(f"Range ##{i}", mn, mx, (mn, mx), key=f"fval_{i}")
            else:
                min_d = df_orig[col_choice].min()
                max_d = df_orig[col_choice].max()
                val = st.date_input(f"Date range ##{i}", value=(min_d, max_d), key=f"fval_{i}")
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    val = tuple(val)
                else:
                    val = (min_d, max_d)
            filters.append({"col": col_choice, "kind": kind, "val": val})

df = apply_filters(df_orig, filters)
st.sidebar.caption(f"Rows after filtering: **{len(df):,}** / {len(df_orig):,}")

# ── sidebar aggregation ───────────────────────────────────────────────────────
st.sidebar.header("Aggregation")
groupby_col = st.sidebar.selectbox("Group by", ["(none)"] + all_cols, key="groupby_col")
agg_fn = st.sidebar.selectbox("Aggregation function", ["sum", "mean", "count", "min", "max"], key="agg_fn")

if groupby_col != "(none)":
    num_cols = df.select_dtypes("number").columns.tolist()
    if num_cols:
        df_agg = df.groupby(groupby_col, dropna=False)[num_cols].agg(agg_fn).reset_index()
    else:
        df_agg = df.groupby(groupby_col, dropna=False).size().reset_index(name="count")
else:
    df_agg = None

# ── chart builder ─────────────────────────────────────────────────────────────
st.subheader("Chart Builder")

chart_df = df_agg if df_agg is not None else df
all_chart_cols = list(chart_df.columns)
num_chart_cols = chart_df.select_dtypes("number").columns.tolist()
cat_chart_cols = chart_df.select_dtypes(exclude="number").columns.tolist()
dt_cols = [c for c in chart_df.columns if pd.api.types.is_datetime64_any_dtype(chart_df[c])]

left, right = st.columns([1, 2])

with left:
    chart_type = st.selectbox(
        "Chart type",
        ["Histogram", "Box Plot", "Scatter Plot", "Line Chart", "Bar Chart", "Heatmap", "Violin Plot", "Pie Chart"],
        key="chart_type",
    )

    cfg = {"chart_type": chart_type}

    if chart_type == "Histogram":
        cfg["x"] = st.selectbox("X-axis", num_chart_cols or all_chart_cols, key="h_x")
        cfg["bins"] = st.slider("Bins", 5, 200, 30, key="h_bins")
        color_opt = st.selectbox("Color (optional)", ["(none)"] + cat_chart_cols, key="h_color")
        cfg["color"] = None if color_opt == "(none)" else color_opt

    elif chart_type == "Box Plot":
        cfg["y"] = st.selectbox("Y-axis", num_chart_cols or all_chart_cols, key="bp_y")
        x_opt = st.selectbox("X-axis / grouping (optional)", ["(none)"] + cat_chart_cols, key="bp_x")
        cfg["x"] = None if x_opt == "(none)" else x_opt
        color_opt = st.selectbox("Color (optional)", ["(none)"] + cat_chart_cols, key="bp_color")
        cfg["color"] = None if color_opt == "(none)" else color_opt

    elif chart_type == "Scatter Plot":
        cfg["x"] = st.selectbox("X-axis", num_chart_cols or all_chart_cols, key="sc_x")
        cfg["y"] = st.selectbox("Y-axis", num_chart_cols or all_chart_cols, key="sc_y")
        color_opt = st.selectbox("Color (optional)", ["(none)"] + all_chart_cols, key="sc_color")
        cfg["color"] = None if color_opt == "(none)" else color_opt
        size_opt = st.selectbox("Size (optional)", ["(none)"] + num_chart_cols, key="sc_size")
        cfg["size"] = None if size_opt == "(none)" else size_opt
        cfg["trendline"] = st.checkbox("Show trendline (OLS)", key="sc_trend")

    elif chart_type == "Line Chart":
        x_pref = dt_cols + num_chart_cols + all_chart_cols
        x_pref = list(dict.fromkeys(x_pref))
        cfg["x"] = st.selectbox("X-axis", x_pref, key="lc_x")
        remaining = [c for c in num_chart_cols if c != cfg["x"]] or num_chart_cols
        cfg["y_cols"] = st.multiselect("Y column(s)", remaining, default=remaining[:1], key="lc_y")
        color_opt = st.selectbox("Color (optional)", ["(none)"] + cat_chart_cols, key="lc_color")
        cfg["color"] = None if color_opt == "(none)" else color_opt

    elif chart_type == "Bar Chart":
        cfg["x"] = st.selectbox("X-axis", all_chart_cols, key="bc_x")
        cfg["y"] = st.selectbox("Y-axis", num_chart_cols or all_chart_cols, key="bc_y")
        color_opt = st.selectbox("Color (optional)", ["(none)"] + cat_chart_cols, key="bc_color")
        cfg["color"] = None if color_opt == "(none)" else color_opt
        cfg["orientation"] = st.radio("Orientation", ["v", "h"], key="bc_orient")
        cfg["bar_agg"] = st.selectbox("Aggregation", ["sum", "mean", "count"], key="bc_agg")

    elif chart_type == "Heatmap":
        cfg["corr_matrix"] = st.checkbox("Correlation matrix (numeric cols)", value=True, key="hm_corr")
        if not cfg["corr_matrix"]:
            cfg["x"] = st.selectbox("X-axis (columns)", all_chart_cols, key="hm_x")
            cfg["y"] = st.selectbox("Y-axis (rows)", all_chart_cols, key="hm_y")
            cfg["z"] = st.selectbox("Z-axis (values)", num_chart_cols or all_chart_cols, key="hm_z")

    elif chart_type == "Violin Plot":
        cfg["y"] = st.selectbox("Y-axis", num_chart_cols or all_chart_cols, key="vp_y")
        cfg["x"] = st.selectbox("X-axis / grouping", cat_chart_cols or all_chart_cols, key="vp_x")
        color_opt = st.selectbox("Color (optional)", ["(none)"] + cat_chart_cols, key="vp_color")
        cfg["color"] = None if color_opt == "(none)" else color_opt

    elif chart_type == "Pie Chart":
        cfg["values"] = st.selectbox("Values", num_chart_cols or all_chart_cols, key="pc_val")
        cfg["names"] = st.selectbox("Names", cat_chart_cols or all_chart_cols, key="pc_names")
        cfg["top_n"] = st.slider("Top N", 5, 20, 10, key="pc_topn")

    generate = st.button("Generate Chart", type="primary", use_container_width=True)

with right:
    fig = None
    if generate or st.session_state.get("last_cfg") == cfg:
        fig = build_chart(chart_df, cfg)
        if fig:
            st.session_state["last_cfg"] = cfg
            st.session_state["last_fig"] = fig

    if "last_fig" in st.session_state and st.session_state["last_fig"] is not None:
        st.plotly_chart(st.session_state["last_fig"], use_container_width=True)

        if st.button("Save Chart", use_container_width=True):
            st.session_state.saved_charts.append(
                {"cfg": dict(cfg), "chart_df": chart_df.copy()}
            )
            st.success(f"Chart saved! ({len(st.session_state.saved_charts)} total)")

# ── summary stats panel ───────────────────────────────────────────────────────
st.subheader("Summary Stats")

if df_agg is not None:
    st.write(f"Grouped by **{groupby_col}** — aggregation: **{agg_fn}**")
    st.dataframe(df_agg, use_container_width=True)
    csv_data = df_agg.to_csv(index=False).encode()
    st.download_button("Download aggregated data as CSV", csv_data, "aggregated.csv", "text/csv")
else:
    st.write(f"Filtered rows: **{len(df):,}**")
    st.dataframe(df.describe(include="all"), use_container_width=True)
    csv_data = df.to_csv(index=False).encode()
    st.download_button("Download filtered data as CSV", csv_data, "filtered.csv", "text/csv")

# ── saved charts gallery ──────────────────────────────────────────────────────
if st.session_state.saved_charts:
    st.subheader(f"Saved Charts Gallery ({len(st.session_state.saved_charts)})")
    cols_per_row = 2
    for i, saved in enumerate(st.session_state.saved_charts):
        if i % cols_per_row == 0:
            gallery_cols = st.columns(cols_per_row)
        with gallery_cols[i % cols_per_row]:
            st.caption(f"Chart {i + 1} — {saved['cfg']['chart_type']}")
            fig_saved = build_chart(saved["chart_df"], saved["cfg"])
            if fig_saved:
                st.plotly_chart(fig_saved, use_container_width=True)

    if st.button("Clear saved charts"):
        st.session_state.saved_charts = []
        st.rerun()
