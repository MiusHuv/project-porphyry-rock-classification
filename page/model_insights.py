# File: page/model_insights.py
# -----------------------------------------------------------------------------
import streamlit as st
from PIL import Image
import os
import pandas as pd # Import pandas for dummy DataFrame
import numpy as np  # Import numpy for dummy DataFrame
from core.model_loader import load_selected_model, load_preprocessor_object
from core.visualizer import get_feature_importances, plot_feature_importances_func
from core.data_handler import EXPECTED_COLUMNS
from util.language import T # Removed TEXTS

def show_page():
    st.title(T("model_insights_title", default="Model Interpretability Insights"))
    st.markdown(T("model_insights_intro", default="""
        This section provides insights into what features the models consider important.
        Due to computational constraints in a live app, SHAP plots are typically shown as
        pre-generated static images from the model training phase. Feature importances for
        tree-based models and linear SVM are generated based on the selected pre-trained model.
    """))

    model_options = ["Random Forest", "XGBoost", "SVM", "DNN-Keras"] # These could be keys if model names need translation
    selected_model_name = st.selectbox(
        T("model_insights_model_select_label", default="Select a model to view insights:"),
        model_options,
        key="insights_model_select"
    )

    st.subheader(T("model_insights_subheader_for_model", model_name=selected_model_name, default=f"Insights for: {selected_model_name}"))

    # --- Feature Importances (for non-DNN) ---
    if selected_model_name != "DNN-Keras":
        st.markdown(T("model_insights_fi_header", default="#### Feature Importances"))
        
        @st.cache_resource
        def get_preprocessor_insights():
            return load_preprocessor_object()
        preprocessor = get_preprocessor_insights()

        @st.cache_resource
        def get_model_insights(model_name):
            return load_selected_model(model_name)
        model = get_model_insights(selected_model_name)

        if model and preprocessor:
            try:
                dummy_input = pd.DataFrame(np.zeros((1, len(EXPECTED_COLUMNS))), columns=EXPECTED_COLUMNS)
                if hasattr(preprocessor, 'get_feature_names_out'):
                    transformed_feature_names = preprocessor.get_feature_names_out(input_features=EXPECTED_COLUMNS)
                else: 
                    transformed_feature_names = EXPECTED_COLUMNS
            except Exception as e:
                st.warning(T("model_insights_fi_warning_feature_names", error_message=str(e), default=f"Could not reliably determine transformed feature names; using original. Error: {e}"))
                transformed_feature_names = EXPECTED_COLUMNS

            importances = get_feature_importances(model, transformed_feature_names, selected_model_name)
            if importances is not None and not importances.empty:
                fig = plot_feature_importances_func(importances, selected_model_name) # Assumes this function handles its own internal text or is language-agnostic
                if fig:
                    st.pyplot(fig)
                    st.caption(T("model_insights_fi_plot_caption", default="Bar chart showing the relative importance of features for the selected model. Higher scores indicate greater influence."))
                else:
                    st.info(T("model_insights_fi_plot_error", default="Feature importances could not be plotted for this model type/configuration."))
            elif selected_model_name == "SVM" and (not hasattr(model, 'coef_') or model.kernel != 'linear'):
                 st.info(T("model_insights_fi_svm_info", default="Feature importances for SVM are typically shown for linear kernels via coefficients."))
            elif importances is None or importances.empty : # Catch cases where get_feature_importances returns None or empty
                st.info(T("model_insights_fi_unavailable_info", model_name=selected_model_name, default=f"Feature importances are not directly available or easily extractable for {selected_model_name} in this app."))
        else:
            st.warning(T("model_insights_fi_load_warning", model_name=selected_model_name, default=f"Could not load model or preprocessor for {selected_model_name} to show feature importances."))
            
    else: # DNN-Keras
        st.info(T("model_insights_fi_dnn_info", default="Feature importances for Deep Neural Networks are complex and often explored using techniques like SHAP, which are presented as static plots here."))

    # --- SHAP Plots (Static Images) ---
    st.markdown(T("model_insights_shap_summary_header", default="#### SHAP Summary Plot (Example from Training)"))
    st.write(T("model_insights_shap_summary_desc", default="""
        SHAP (SHapley Additive exPlanations) values interpret the impact of each feature on individual predictions.
        The summary plot below is an example from the model's training phase, showing feature importance
        and the distribution of SHAP values for each feature.
    """))
    
    shap_image_base_path = "assets/shap_plots" # Define a base path for SHAP images
    shap_summary_image_filename = f"shap_summary_{selected_model_name.lower().replace(' ', '_').replace('-', '_')}.png"
    shap_summary_image_path = os.path.join(shap_image_base_path, shap_summary_image_filename)

    if os.path.exists(shap_summary_image_path):
        try:
            image = Image.open(shap_summary_image_path)
            st.image(image, caption=T("model_insights_shap_summary_caption", model_name=selected_model_name, default=f"SHAP Summary Plot for {selected_model_name} (from training data)."))
        except Exception as e:
            st.warning(T("model_insights_shap_image_load_warning", image_type="Summary", error_message=str(e), default=f"Could not load SHAP Summary image: {e}"))
    else:
        st.info(T("model_insights_shap_image_not_found", image_type="Summary", model_name=selected_model_name, image_path=shap_summary_image_path, default=f"No pre-generated SHAP Summary plot found for {selected_model_name} at '{shap_summary_image_path}'."))

    st.markdown(T("model_insights_shap_dependence_header", default="#### SHAP Dependence Plots (Examples from Training)"))
    st.write(T("model_insights_shap_dependence_desc", default="""
        SHAP dependence plots show how a single feature's value affects the SHAP value (and thus the prediction),
        potentially highlighting interactions with another feature. These are examples from the training phase.
    """))
    
    # Example for two dependence plots (adjust feature names and number as needed)
    # You should determine your top features for dependence plots from your analysis
    top_features_for_dependence = { 
        "Random Forest": ["SiO2", "Cu"], # Example features
        "XGBoost": ["Al2O3", "Au"],   # Example features
        "SVM": ["MgO", "K2O"],        # Example features for SVM if applicable
        "DNN-Keras": ["Fe2O3T", "S"]  # Example features for DNN
    }

    if selected_model_name in top_features_for_dependence:
        features_to_plot = top_features_for_dependence[selected_model_name][:2] # Take first two
        for i, feature_name in enumerate(features_to_plot):
            shap_dep_image_filename = f"shap_dependence_{selected_model_name.lower().replace(' ', '_').replace('-', '_')}_{feature_name.lower()}.png"
            shap_dep_image_path = os.path.join(shap_image_base_path, shap_dep_image_filename)
            
            if os.path.exists(shap_dep_image_path):
                try:
                    image_dep = Image.open(shap_dep_image_path)
                    st.image(image_dep, caption=T("model_insights_shap_dependence_caption", feature_name=feature_name, model_name=selected_model_name, default=f"SHAP Dependence Plot for {feature_name} ({selected_model_name}, from training)."))
                except Exception as e:
                    st.warning(T("model_insights_shap_image_load_warning", image_type=f"Dependence ({feature_name})", error_message=str(e), default=f"Could not load SHAP Dependence ({feature_name}) image: {e}"))
            else:
                st.info(T("model_insights_shap_image_not_found", image_type=f"Dependence ({feature_name})", model_name=selected_model_name, image_path=shap_dep_image_path, default=f"No pre-generated SHAP Dependence plot for {feature_name} ({selected_model_name}) found at '{shap_dep_image_path}'."))
    else:
        st.info(T("model_insights_shap_dependence_config_missing", model_name=selected_model_name, default=f"Configuration for SHAP dependence plots not available for {selected_model_name}."))


    st.markdown("---")
    st.subheader(T("model_insights_geological_meaning_header", default="Geological Meaning of Influential Variables (Discussion)"))
    st.markdown(T("model_insights_geological_meaning_desc", default="""
    *(This section should be filled with your discussion of the 5-10 most influential variables
    and their geological meaning, based on your project's interpretability analysis.)*

    For example:
    * **SiO₂ (Silica):** Higher silica content often correlates with more felsic magmas, which can be associated with certain types of porphyry deposits...
    * **Cu/Au Ratio (if engineered):** This directly informs the classification...
    * **K₂O/Na₂O:** Indicates alkalinity, which plays a role in magma evolution and mineralization...
    * **Sr/Y Ratio:** Can be an indicator of slab melt involvement or crustal thickness...
    """))

if __name__ == "__main__":
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    # Create dummy asset files for testing if they don't exist
    dummy_shap_path = "assets/shap_plots"
    if not os.path.exists(dummy_shap_path):
        os.makedirs(dummy_shap_path)
    
    model_options_test = ["Random Forest", "XGBoost", "SVM", "DNN-Keras"]
    for model_n in model_options_test:
        model_n_safe = model_n.lower().replace(' ', '_').replace('-', '_')
        # Dummy summary plot
        summary_file = os.path.join(dummy_shap_path, f"shap_summary_{model_n_safe}.png")
        if not os.path.exists(summary_file):
            try:
                Image.new('RGB', (100, 30), color = 'red').save(summary_file)
            except ImportError: pass # Pillow might not be installed in a very basic env
        # Dummy dependence plots
        # features_to_plot_test = top_features_for_dependence.get(model_n, [])[:2]
        # for feat in features_to_plot_test:
        #     dep_file = os.path.join(dummy_shap_path, f"shap_dependence_{model_n_safe}_{feat.lower()}.png")
        #     if not os.path.exists(dep_file):
        #         try: Image.new('RGB', (100, 30), color = 'blue').save(dep_file)
        #         except ImportError: pass


    show_page()