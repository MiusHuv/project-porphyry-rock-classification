# File: page/help_about.py
# -----------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from util.language import T

def show_page():
    st.title(T("help_about_title", default="Help / About This Tool"))

    st.subheader(T("help_about_usage_header", default="Usage Instructions"))
    st.markdown(T("help_about_usage_content", default="""
    1.  **Navigate to 'Run Prediction'**: Use the sidebar to go to the prediction page.
    2.  **Upload Data**:
        * Click 'Browse files' to upload your sample data.
        * The file must be in `.csv` or `.xlsx` format.
        * Ensure your data contains the 36 required geochemical features (see 'Feature Descriptions' below). Column names should match.
        * A `sample_data.csv` file is available in the `assets/` directory of the project for reference.
    3.  **Select Model**: Choose one of the pre-trained models (Random Forest, XGBoost, SVM, DNN-Keras) from the sidebar.
    4.  **Run Prediction**:
        * If your data includes a column with true labels (e.g., 'Actual_Class'), you can check the box "My data includes a 'True Label' column..." and select that column. This enables performance visualization.
        * Click the 'Predict using [Selected Model]' button.
    5.  **View Results**: Predictions and probabilities will be displayed in a table. You can download these results as a CSV file.
    6.  **Visualize Performance (Optional)**: If you provided a true label column, navigate to the 'Performance Visualizer' page to see the confusion matrix and ROC curve for the predictions made.
    7.  **Explore Model Insights**: Navigate to 'Model Insights' to see feature importances (for some models) and example SHAP plots (static images from training).
    """))

    st.subheader(T("help_about_features_header", default="Feature Descriptions (Input Data Requirements)"))
    st.markdown(T("help_about_features_content", default="""
    Your input data must contain the following 36 major- and trace-element features:
    * **Major Elements (wt %):** SiO₂, TiO₂, Al₂O₃, TFe₂O₃, MnO, MgO, CaO, Na₂O, K₂O, P₂O₅
    * **Trace Elements (ppm):** Rb, Sr, Y, Zr, Nb, Ba, La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, Th, U
    * Missing values in these features will be imputed with zero before further processing (as per project CLR handling).
    """))

    st.subheader(T("help_about_model_info_header", default="Model Information"))
    st.markdown(T("help_about_model_info_content", default="""
    This tool uses four types of pre-trained machine learning models:
    * **Random Forest:** An ensemble learning method using multiple decision trees.
    * **XGBoost:** A gradient boosting framework, known for high performance.
    * **SVM (Support Vector Machine):** A classifier that finds an optimal hyperplane to separate classes.
    * **DNN-Keras:** A Deep Neural Network built using the Keras API.
    Each model was trained on the "2025-Project-Data.xlsx" dataset.
    """))

    st.subheader(T("help_about_tech_stack_header", default="Library Versions & Tech Stack"))
    st.markdown(T("help_about_tech_stack_intro", default="This application was built using Python and relies on several key libraries:"))
    # The f-string content itself is mostly library names and versions, which are not typically translated.
    # The surrounding text is handled by the markdown above and caption below.
    st.code(f"""
    * Streamlit: For the web GUI (version {st.__version__})
    * Pandas: For data manipulation (version {pd.__version__})
    * Numpy: For numerical operations (version {np.__version__})
    * Scikit-learn: For ML models (RF, SVM) and metrics (version {sklearn.__version__})
    * XGBoost: For the XGBoost model (version {xgb.__version__ if 'xgb' in globals() else 'N/A'})
    * Matplotlib & Seaborn: For plotting (versions {matplotlib.__version__}, {sns.__version__})
    * Joblib: For model persistence (version {joblib.__version__})
    """)
    st.caption(T("help_about_tech_stack_caption", default="To get exact versions, you can typically use `pip freeze > requirements.txt` in your project's virtual environment and list key ones here."))

    st.subheader(T("help_about_project_ack_header", default="Project & Acknowledgements"))
    st.markdown(T("help_about_project_ack_content", default="""
    * This tool was developed as part of the COMMON TOOLS FOR DATA SCIENCE final project.
    * **Team:** [Your Team Name or Group Number: e.g., 2025GXX-ProjectName]
        * [Team Member 1]
        * [Team Member 2]
        * [Team Member 3]
        * [Team Member 4 (and 5 if applicable)]
    * **Dataset:** "2025-Project-Data.xlsx" (provided for the course).
    * **Inspiration for GUI Structure:** `rascore.streamlit.app` by Mitch Parker.
    """))
    
    st.markdown("---")
    st.info(T("help_about_contact_info", default="For issues or questions, please refer to the project documentation or contact the development team."))

if __name__ == "__main__":
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    show_page()