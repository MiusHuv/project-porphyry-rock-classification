# app.py
import streamlit as st

# Optional: Set global page configuration (do this only once, preferably here or in .streamlit/config.toml)
# st.set_page_config(
#     page_title="GeoClassifier",
#     page_icon="⛏️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

st.sidebar.success("Select a page above.")

st.title("Welcome to the Geochemical Classifier! ")
st.markdown("""
This application allows you to classify porphyry rock samples as Cu-rich or Au-rich
based on their elemental composition using pre-trained machine learning models.

**Navigate using the sidebar to:**
- **Home:** You are here.
- **Run Prediction:** Upload your data and get classifications.
- **Performance Visualizer:** Evaluate model performance if you have true labels.
- **Model Insights:** View general interpretability plots for the models.
- **Help/About:** Get usage instructions and app details.
""")

# To make Streamlit recognize the 'pages' directory for multi-page apps,
# this file (app.py) and the 'pages' directory must be in the root of your Streamlit app's execution path.
# No other specific code is needed here for page navigation if using the 'pages/' folder.
