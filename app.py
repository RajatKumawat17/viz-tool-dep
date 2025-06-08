# File: app.py
import base64
import streamlit as st
import os
import tempfile
from llm_viz_engine import LLMVisualizationEngine

st.set_page_config(page_title="LLM Visualization Tool", layout="wide")
st.title("📊 LLM-Powered Data Visualization")
st.markdown("Upload a CSV file and get intelligent insights and visualizations.")

# GROQ key input
user_key = st.text_input("Enter your Groq API Key (Optional)", type="password")

def init_engine():
    try:
        return LLMVisualizationEngine(groq_api_key=user_key if user_key else None)
    except Exception as e:
        st.error(f"LLM engine init failed: {e}")
        return None

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Prompt context input
prompt_context = st.text_area("Optional: Add context about your dataset (e.g., goal of analysis)", "")

# Prompt for custom graph
user_prompt = st.text_input("Optional: Enter a custom graph description or prompt", "Show correlation between all numerical features")

if uploaded_file:
    engine = init_engine()
    if not engine:
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    with st.spinner("Analyzing your data..."):
        try:
            engine.load_data(tmp_path)
            engine.create_derived_features()
            engine.profile_data()
            engine.detect_data_context()
            engine.generate_insights()

            st.subheader("🔍 Insights")
            for insight in engine.insights:
                st.markdown(f"- {insight}")

            st.subheader("📊 Suggested Visualizations")
            viz_suggestions = engine.suggest_visualizations()
            for viz_config in viz_suggestions[:4]:
                chart_path = engine.create_visualization(viz_config)
                if chart_path and os.path.exists(chart_path):
                    st.markdown(f"**{viz_config['title']}**")
                    st.image(chart_path, caption=viz_config['description'])

            if user_prompt.strip():
                st.subheader("✨ Custom Prompt-based Visualization")
                full_prompt = f"{prompt_context}\n{user_prompt}" if prompt_context else user_prompt
                chart_path = engine.visualize_from_prompt(full_prompt)
                if chart_path and os.path.exists(chart_path):
                    st.image(chart_path, caption=f"Prompt: {user_prompt}")
                else:
                    st.info("Could not generate visualization from prompt.")

            # PDF Report Generation
            pdf_path = engine.generate_pdf_report()
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download PDF Report", f, file_name="data_analysis_report.pdf")
            # HTML Report Download
            html_content = engine.generate_html_report()
            b64_html = base64.b64encode(html_content.encode('utf-8')).decode()
            st.download_button("⬇️ Download HTML Report", data=html_content, file_name="report.html", mime="text/html")


        except Exception as e:
            st.error(f"Error: {str(e)}")

    os.remove(tmp_path)
