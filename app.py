import streamlit as st

st.set_page_config(
    page_title="DataWrangler Pro",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

defaults = {
    "df": None,
    "original_df": None,
    "transform_log": [],
    "filename": None,
    "df_history": [],
    "validation_rules": [],
    "saved_charts": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

pages = [
    st.Page("pages/a_upload.py", title="Upload & Overview", icon="📁"),
    st.Page("pages/b_cleaning.py", title="Cleaning Studio", icon="🧹"),
    st.Page("pages/c_visualization.py", title="Visualization Builder", icon="📊"),
    st.Page("pages/d_export.py", title="Export & Report", icon="📤"),
]

pg = st.navigation(pages)

with st.sidebar:
    st.markdown("---")
    st.markdown("### Dataset Info")
    if st.session_state.df is not None:
        st.markdown(f"**File:** {st.session_state.filename}")
        rows, cols = st.session_state.df.shape
        st.markdown(f"**Shape:** {rows:,} rows × {cols} cols")
        st.markdown(f"**Steps applied:** {len(st.session_state.transform_log)}")
    else:
        st.markdown("_No dataset loaded_")

pg.run()
