# ./app.py
import streamlit as st

from page import Home, run_prediction, performance_visualizer, model_insights, help_about

# -----------------------------------------------------------------------------
# 2. SETUP PAGE DICTIONARY (mapping to imported module functions)
# -----------------------------------------------------------------------------
PAGES = {
    "Home": Home.show_page,
    "Run Prediction": run_prediction.show_page, 
    "Performance Visualizer": performance_visualizer.show_page,
    "Model Insights": model_insights.show_page,
    "Help / About": help_about.show_page
}

# Initialize session state
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

# -----------------------------------------------------------------------------
# 3. CREATE THE SIDEBAR FOR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("Main Menu")

def update_page():
    st.session_state.selected_page = st.session_state.page_selector

selection = st.sidebar.selectbox(
    "Select Page",
    options=list(PAGES.keys()),
    key="page_selector",
    on_change=update_page,
    index=list(PAGES.keys()).index(st.session_state.selected_page)
)

# -----------------------------------------------------------------------------
# 4. DISPLAY THE SELECTED PAGE
# -----------------------------------------------------------------------------
page_function = PAGES[st.session_state.selected_page]
page_function()

# Initialize other session state variables
# (Same as in Option 1's app.py)
