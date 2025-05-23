# ./page/Home.py
import streamlit as st
from PIL import Image 
from pathlib import Path

def show_page():

    left_col, right_col = st.columns(2)

    img_path = Path(__file__).parent.parent / "assets" / "Sample.png"

    try:
        img = Image.open(img_path)
        with left_col:
            st.image(img, caption="Porphyry Deposit Illustration", use_container_width=True)
    except FileNotFoundError:
        with left_col:
            st.warning("Homepage image 'Sample.png' not found.")
    
    with left_col: # This ensures the markdown is always displayed, even if image fails
        st.markdown("""
            <style>
                .stApp {
                    background-color: #f0f0f5;  /* Light gray background */
                }
            </style>
            """, unsafe_allow_html=True)

    with right_col:
        st.title("Welcome to the Geochemical Rock Classifier")
        st.write("This tool classifies porphyry rock samples based on their elemental composition.")


    st.markdown("""
        This tool classifies porphyry rock samples as **Cu-rich** or **Au-rich** based on their
        elemental composition.

        **How to Use:**
        1. Navigate to the **Run Prediction** page using the sidebar.
        2. Upload your sample data (CSV or XLSX format).
        3. Select a pre-trained classification model.
        4. View the predictions and probabilities.
        5. Optionally, if your data includes true labels, use the **Performance Visualizer**.

        The models were trained on the "2025-Project-Data.xlsx" dataset, comprising
        36 major- and trace-element features.

        For more detailed instructions and information about the models and features,
        please visit the **Help / About** page.
    """)


    # You could add an image from your assets folder:
    # from PIL import Image
    # try:
    #     image = Image.open("assets/your_geology_image.png")
    #     st.image(image, caption="Porphyry Deposit Illustration")
    # except FileNotFoundError:
    #     st.warning("Homepage image not found.")
