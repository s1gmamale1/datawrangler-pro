import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import numpy as np

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


def build_chart_mpl(df, cfg):
    """Build chart using matplotlib. Returns a Figure or None."""
    ctype = cfg["chart_type"]
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if ctype == "Histogram":
            col = cfg["x"]
            bins = cfg.get("bins", 30)
            if cfg.get("color"):
                groups = df.groupby(cfg["color"])[col]
                for name, group in groups:
                    ax.hist(group.dropna(), bins=bins, alpha=0.6, label=str(name))
                ax.legend()
            else:
                ax.hist(df[col].dropna(), bins=bins, edgecolor="black", alpha=0.7)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.set_title(f"Histogram of {col}")

        elif ctype == "Box Plot":
            y_col = cfg["y"]
            if cfg.get("x"):
                groups = df.groupby(cfg["x"])[y_col].apply(lambda g: g.dropna().tolist())
                labels = [str(l) for l in groups.index]
                ax.boxplot(groups.values, labels=labels)
                ax.set_xlabel(cfg["x"])
            else:
                ax.boxplot(df[y_col].dropna())
            ax.set_ylabel(y_col)
            ax.set_title(f"Box Plot of {y_col}")
            plt.xticks(rotation=45, ha="right")

        elif ctype == "Scatter Plot":
            x_col, y_col = cfg["x"], cfg["y"]
            if cfg.get("color"):
                groups = df.groupby(cfg["color"])
                for name, group in groups:
                    ax.scatter(group[x_col], group[y_col], alpha=0.6, label=str(name),
                               s=group[cfg["size"]] if cfg.get("size") else 30)
                ax.legend()
            else:
                s = df[cfg["size"]] if cfg.get("size") else 30
                ax.scatter(df[x_col], df[y_col], alpha=0.6, s=s)
            if cfg.get("trendline"):
                mask = df[[x_col, y_col]].dropna()
                if len(mask) > 1:
                    z = np.polyfit(mask[x_col], mask[y_col], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(mask[x_col].min(), mask[x_col].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", linewidth=2, label="OLS trendline")
                    ax.legend()
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter: {x_col} vs {y_col}")

        elif ctype == "Line Chart":
            x_col = cfg["x"]
            y_cols = cfg.get("y_cols", [])
            for yc in y_cols:
                ax.plot(df[x_col], df[yc], label=yc, marker="." if len(df) < 200 else None)
            if len(y_cols) > 1:
                ax.legend()
            ax.set_xlabel(x_col)
            ax.set_ylabel(", ".join(y_cols))
            ax.set_title("Line Chart")
            plt.xticks(rotation=45, ha="right")

        elif ctype == "Bar Chart":
            agg_fn = cfg.get("bar_agg", "sum")
            x_col, y_col = cfg["x"], cfg["y"]
            agg_map = {"sum": "sum", "mean": "mean", "count": "count"}
            grp_cols = [c for c in [x_col, cfg.get("color")] if c]
            if y_col in grp_cols:
                grp_cols = [x_col]
            dfa = df.groupby(grp_cols, dropna=False)[y_col].agg(agg_map[agg_fn]).reset_index()

            if cfg.get("color") and cfg["color"] in dfa.columns:
                pivot = dfa.pivot_table(index=x_col, columns=cfg["color"], values=y_col, fill_value=0)
                pivot.plot(kind="barh" if cfg.get("orientation") == "h" else "bar", ax=ax, edgecolor="black")
            else:
                if cfg.get("orientation") == "h":
                    ax.barh(dfa[x_col].astype(str), dfa[y_col], edgecolor="black")
                    ax.set_xlabel(y_col)
                    ax.set_ylabel(x_col)
                else:
                    ax.bar(dfa[x_col].astype(str), dfa[y_col], edgecolor="black")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
            ax.set_title(f"Bar Chart — {agg_fn}({y_col})")
            plt.xticks(rotation=45, ha="right")

        elif ctype == "Heatmap":
            plt.close(fig)
            if cfg.get("corr_matrix"):
                num_df = df.select_dtypes("number")
                corr = num_df.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
                ax.set_title("Correlation Matrix")
            else:
                pivot = df.pivot_table(index=cfg["y"], columns=cfg["x"], values=cfg["z"], aggfunc="mean")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", ax=ax)
                ax.set_title("Heatmap")

        elif ctype == "Violin Plot":
            y_col = cfg["y"]
            x_col = cfg.get("x")
            if x_col:
                categories = df[x_col].dropna().unique()
                data = [df[df[x_col] == cat][y_col].dropna().values for cat in categories]
                parts = ax.violinplot(data, showmeans=True, showmedians=True)
                ax.set_xticks(range(1, len(categories) + 1))
                ax.set_xticklabels([str(c) for c in categories], rotation=45, ha="right")
                ax.set_xlabel(x_col)
            else:
                ax.violinplot(df[y_col].dropna().values, showmeans=True, showmedians=True)
            ax.set_ylabel(y_col)
            ax.set_title(f"Violin Plot of {y_col}")

        elif ctype == "Pie Chart":
            top_n = cfg.get("top_n", 10)
            dfa = df.groupby(cfg["names"])[cfg["values"]].sum().nlargest(top_n)
            ax.pie(dfa.values, labels=dfa.index, autopct="%1.1f%%", startangle=90)
            ax.set_title(f"Pie Chart — Top {top_n}")

        fig.tight_layout()
        return fig

    except Exception as e:
        plt.close(fig)
        st.error(f"Matplotlib chart error: {e}")
        return None


# ── sidebar rendering engine ──────────────────────────────────────────────────
st.sidebar.header("Rendering Engine")
render_engine = st.sidebar.radio("Library", ["Plotly (interactive)", "Matplotlib (static)"], key="render_engine")

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
    if generate:
        st.session_state["last_cfg"] = dict(cfg)

    active_cfg = st.session_state.get("last_cfg")
    if active_cfg:
        use_mpl = render_engine.startswith("Matplotlib")
        if use_mpl:
            fig = build_chart_mpl(chart_df, active_cfg)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            fig = build_chart(chart_df, active_cfg)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"main_{active_cfg.get('chart_type','chart')}")

        if st.button("Save Chart", use_container_width=True):
            st.session_state.saved_charts.append(
                {"cfg": dict(cfg), "chart_df": chart_df.copy(), "engine": render_engine}
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
            saved_engine = saved.get("engine", "Plotly (interactive)")
            st.caption(f"Chart {i + 1} — {saved['cfg']['chart_type']} ({saved_engine.split(' ')[0]})")
            if saved_engine.startswith("Matplotlib"):
                fig_saved = build_chart_mpl(saved["chart_df"], saved["cfg"])
                if fig_saved:
                    st.pyplot(fig_saved, use_container_width=True)
                    plt.close(fig_saved)
            else:
                fig_saved = build_chart(saved["chart_df"], saved["cfg"])
                if fig_saved:
                    st.plotly_chart(fig_saved, use_container_width=True, key=f"saved_chart_{i}")

    if st.button("Clear saved charts"):
        st.session_state.saved_charts = []
        st.rerun()
