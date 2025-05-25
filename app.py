# ./app.py
import streamlit as st

from page import Home, run_prediction, performance_visualizer, model_insights, help_about, data_analysis
from pathlib import Path
from util.language import T, TEXTS
# -----------------------------------------------------------------------------
# 1. INITIALIZE SESSION STATE FOR LANGUAGE
# -----------------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"  # Default language

# Function to update language
def update_language():
    st.session_state.lang = st.session_state.language_selector

# -----------------------------------------------------------------------------
# 2. SETUP PAGE DICTIONARY (mapping to imported module functions)
# -----------------------------------------------------------------------------
# Use static keys for PAGES, display names will be translated
PAGES_CONFIG = {
    "Home": {"func": Home.show_page, "icon": "🏠", "key": "page_home"},
    "Data Analysis": {"func": data_analysis.show_page, "icon": "📊", "key": "page_data_analysis"},
    "Run Prediction": {"func": run_prediction.show_page, "icon": "🚀", "key": "page_run_prediction"},
    "Performance Visualizer": {"func": performance_visualizer.show_page, "icon": "📈", "key": "page_performance_visualizer"},
    "Model Insights": {"func": model_insights.show_page, "icon": "💡", "key": "page_model_insights"},
    "Help / About": {"func": help_about.show_page, "icon": "❓", "key": "page_help_about"}
}

if "selected_page_key" not in st.session_state:
    st.session_state.selected_page_key = "Home" # Default to the key of the home page

# -----------------------------------------------------------------------------
# 3. CREATE THE SIDEBAR FOR NAVIGATION
# -----------------------------------------------------------------------------

# Language Selector
st.sidebar.radio(
    label=T("language_label"), # "Language"
    options=["en", "zh"],
    format_func=lambda lang_code: "English" if lang_code == "en" else "中文",
    key="language_selector",
    on_change=update_language,
    horizontal=True,
)
st.sidebar.markdown("---")


st.sidebar.title(T("main_menu_title")) # "Main Menu"

# Create display names for the selectbox based on current language
page_options_keys = list(PAGES_CONFIG.keys())
page_display_names = [f"{PAGES_CONFIG[key]['icon']} {T(PAGES_CONFIG[key]['key'])}" for key in page_options_keys]


def update_page_selection():
    # Find the key corresponding to the selected display name
    selected_display_name = st.session_state.page_selector_display
    for key, config in PAGES_CONFIG.items():
        if f"{config['icon']} {T(config['key'])}" == selected_display_name:
            st.session_state.selected_page_key = key
            break

# Determine the current index for the selectbox
try:
    current_selected_display_name = f"{PAGES_CONFIG[st.session_state.selected_page_key]['icon']} {T(PAGES_CONFIG[st.session_state.selected_page_key]['key'])}"
    current_index = page_display_names.index(current_selected_display_name)
except (KeyError, ValueError):
    current_index = 0 # Default to first page if current selection is somehow invalid


selection_display = st.sidebar.selectbox(
    label=T("select_page_label"), # "Select Page"
    options=page_display_names,
    key="page_selector_display",
    on_change=update_page_selection,
    index=current_index
)

# -----------------------------------------------------------------------------
# 4. DISPLAY THE SELECTED PAGE
# -----------------------------------------------------------------------------
page_function_to_call = PAGES_CONFIG[st.session_state.selected_page_key]["func"]
page_function_to_call()


# Initialize other session state variables if needed
# (Example from your original code, ensure they don't conflict or are managed appropriately)
# if "raw_df" not in st.session_state:
#     st.session_state.raw_df = None
# if "predictions_df" not in st.session_state:
#     st.session_state.predictions_df = None
# if "selected_model_name" not in st.session_state:
#     st.session_state.selected_model_name = None
# if "true_label_column" not in st.session_state:
#     st.session_state.true_label_column = None
# if "show_performance" not in st.session_state:
#     st.session_state.show_performance = False