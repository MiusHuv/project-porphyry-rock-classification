# ./app.py
import streamlit as st

from page import Home, run_prediction, performance_visualizer, model_insights, help_about, data_analysis
from pathlib import Path
from util.language import T, TEXTS
# -----------------------------------------------------------------------------
# 1. INITIALIZE SESSION STATE FOR LANGUAGE
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide")

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
language_options_map = {"en": "English", "zh": "中文"}
language_option_codes = list(language_options_map.keys()) # ["en", "zh"]

# Determine the current index for the selectbox based on st.session_state.lang
current_lang_code = st.session_state.get("lang", "en")
try:
    current_lang_index = language_option_codes.index(current_lang_code)
except ValueError:
    current_lang_index = 0 # Default to the first option (English)

st.sidebar.selectbox(
    label=T("language_label"), # "Language" / "语言"
    options=language_option_codes,
    format_func=lambda lang_code: language_options_map[lang_code], # Displays "English" or "中文"
    key="lang",  # Directly use st.session_state.lang as the key
    index=current_lang_index, # Set the initially selected item
    # No on_change callback needed if just updating st.session_state.lang
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

