# ./page/Home.py
import streamlit as st
from PIL import Image 
from pathlib import Path
from util.language import T, TEXTS

def show_page():

    left_col, right_col = st.columns(2)

    img_path = Path(__file__).parent.parent / "assets" / "Sample.png"

    with left_col:
        img_path = Path(__file__).parent.parent / "assets" / "Sample.png"
        try:
            img = Image.open(img_path)
            st.image(img, caption=T("home_image_caption"), use_container_width=True)
        except FileNotFoundError:
            st.warning("Homepage image 'Sample.png' not found in assets folder.") # This could also be translated

    with left_col: # This ensures the markdown is always displayed, even if image fails
        st.markdown("""
            <style>
                .stApp {
                    background-color: #f0f0f5;  /* Light gray background */
                }
            </style>
            """, unsafe_allow_html=True)

    with right_col:
        st.title(T("home_title"))
        st.subheader(T("home_subtitle"))
        st.markdown(T("home_intro"))
        st.markdown(f"**{T('home_core_features_header')}**")
        st.markdown(f"*   {T('home_core_feature_1')}")
        st.markdown(f"*   {T('home_core_feature_2')}")
        st.markdown(f"*   {T('home_core_feature_3')}")


    # --- 2. Quick Navigation / Call to Action ---
    st.header(T("home_quick_start_header"))
    col1, col2, col3 = st.columns(3)

    # Store page keys directly for navigation, display names are handled by app.py
    # The button labels themselves should be translated.
    with col1:
        if st.button(T("home_button_run_prediction"), use_container_width=True, help=T("home_help_run_prediction")):
            st.session_state.selected_page_key = "Run Prediction"
            st.rerun()

    with col2:
        if st.button(T("home_button_view_performance"), use_container_width=True, help=T("home_help_view_performance")):
            st.session_state.selected_page_key = "Performance Visualizer"
            st.rerun()

    with col3:
        if st.button(T("home_button_model_insights"), use_container_width=True, help=T("home_help_model_insights")):
            st.session_state.selected_page_key = "Model Insights"
            st.rerun()
            
    st.markdown("---")

    # --- 3. Application Overview ---
    st.header(T("home_app_overview_header"))
    st.markdown(T("home_app_overview_p1"))
    st.markdown(f"1.  {T('home_app_overview_li1')}")
    st.markdown(f"2.  {T('home_app_overview_li2')}")
    st.markdown(f"3.  {T('home_app_overview_li3')}")
    st.markdown(T("home_app_overview_p2"))
    
    st.markdown("---")

    # --- 4. Further Assistance ---
    st.info(T("home_further_assistance_info"))