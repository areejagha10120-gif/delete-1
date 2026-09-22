import streamlit as st
from research_agent import run_research


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="Research",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .report-box {
        background-color: #f7f7f7;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="main-title">AI Research Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter a research topic and let the AI research the web and generate a structured report.'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Research topic
# -----------------------------

topic = st.text_area(
    "Research Topic",
    placeholder=(
        "Example: The impact of artificial intelligence on software engineering"
    ),
    height=120
)


# -----------------------------
# Research button
# -----------------------------

if st.button("Start Research", type="primary"):

    if not topic.strip():
        st.warning("Please enter a research topic first.")

    else:

        with st.spinner("Researching the web and preparing your report..."):

            try:

                report = run_research(topic.strip())

                st.success("Research completed.")

                st.markdown("## Research Report")

                st.markdown(
                    f'<div class="report-box">{report}</div>',
                    unsafe_allow_html=True
                )

                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name="research_report.md",
                    mime="text/markdown"
                )

            except Exception as e:

                st.error(
                    "Something went wrong while generating the report."
                )

                st.code(str(e))
