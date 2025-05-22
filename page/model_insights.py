import streamlit as st
from PIL import Image # For displaying pre-generated images
import os
from core.model_loader import load_selected_model, load_preprocessor_object # To get feature names after preprocessing
from core.visualizer import get_feature_importances, plot_feature_importances_func
from core.data_handler import EXPECTED_COLUMNS # Original feature names

def show_page():
    st.title("💡 Model Interpretability Insights")
    st.markdown("""
        This section provides insights into what features the models consider important.
        Due to computational constraints in a live app, SHAP plots are typically shown as
        pre-generated static images from the model training phase. Feature importances for
        tree-based models and linear SVM are generated based on the selected pre-trained model.
    """)

    model_options = ["Random Forest", "XGBoost", "SVM", "DNN-Keras"]
    selected_model_name = st.selectbox(
        "Select a model to view insights:", 
        model_options,
        key="insights_model_select" # Unique key for this selectbox
    )

    st.subheader(f"Insights for: {selected_model_name}")

    # --- Feature Importances (for non-DNN) ---
    if selected_model_name != "DNN-Keras":
        st.markdown("#### Feature Importances")
        
        # Load the preprocessor to get feature names *after* transformation, if they change
        @st.cache_resource
        def get_preprocessor_insights(): # Use a different cache name or scope if needed
            return load_preprocessor_object()
        preprocessor = get_preprocessor_insights()

        # Load the selected model
        @st.cache_resource
        def get_model_insights(model_name): # Use a different cache name
            return load_selected_model(model_name)
        model = get_model_insights(selected_model_name)

        if model and preprocessor:
            # Get feature names. This depends on your preprocessor.
            # If CLR or other transformations change names/number of features:
            try:
                # Create a dummy DataFrame to pass to the preprocessor to get output feature names
                dummy_df = pd.DataFrame(columns=EXPECTED_COLUMNS) 
                # If preprocessor is a scikit-learn pipeline with named steps:
                if hasattr(preprocessor, 'get_feature_names_out'):
                    # This works if all transformers in pipeline support get_feature_names_out
                    # And assumes the dummy_df structure is compatible with the first step.
                    # You might need to pass a dummy dataframe with one row of zeros.
                    import pandas as pd
                    import numpy as np
                    dummy_input = pd.DataFrame(np.zeros((1, len(EXPECTED_COLUMNS))), columns=EXPECTED_COLUMNS)
                    transformed_feature_names = preprocessor.get_feature_names_out(input_features=EXPECTED_COLUMNS)
                else: # Fallback to original names if preprocessor doesn't provide new ones
                    transformed_feature_names = EXPECTED_COLUMNS
            except Exception as e:
                st.warning(f"Could not reliably determine transformed feature names; using original. Error: {e}")
                transformed_feature_names = EXPECTED_COLUMNS

            importances = get_feature_importances(model, transformed_feature_names, selected_model_name)
            if importances is not None and not importances.empty:
                fig = plot_feature_importances_func(importances, selected_model_name)
                if fig:
                    st.pyplot(fig)
                    st.caption("Bar chart showing the relative importance of features for the selected model. Higher scores indicate greater influence.")
                else:
                    st.info("Feature importances could not be plotted for this model type/configuration.")
            elif selected_model_name == "SVM" and (not hasattr(model, 'coef_') or model.kernel != 'linear'):
                 st.info("Feature importances for SVM are typically shown for linear kernels via coefficients.")
            else:
                st.info(f"Feature importances are not directly available or easily extractable for {selected_model_name} in this app.")
        else:
            st.warning(f"Could not load model or preprocessor for {selected_model_name} to show feature importances.")
            
    else: # DNN-Keras
        st.info("Feature importances for Deep Neural Networks are complex and often explored using techniques like SHAP, which are presented as static plots here.")

    # --- SHAP Plots (Static Images) ---
    st.markdown("#### SHAP Summary Plot (Example)")
    st.write("""
        SHAP (SHapley Additive exPlanations) values interpret the impact of each feature on individual predictions.
        The summary plot below is an example of what a SHAP beeswarm plot might look like, showing feature importance
        and the distribution of SHAP values for each feature.
        **(Note: This is a placeholder/example image. You should replace it with your actual pre-generated SHAP plot.)**
    """)
    
    # Try to load a pre-generated SHAP summary plot for the selected model
    shap_image_path = f"assets/shap_summary_{selected_model_name.lower().replace(' ', '_')}.png"
    if os.path.exists(shap_image_path):
        try:
            image = Image.open(shap_image_path)
            st.image(image, caption=f"Example SHAP Summary Plot for {selected_model_name}")
        except Exception as e:
            st.warning(f"Could not load SHAP summary image: {e}")
    else:
        st.info(f"No pre-generated SHAP summary plot found for {selected_model_name} at '{shap_image_path}'.")

    st.markdown("#### SHAP Dependence Plots (Examples)")
    st.write("""
        SHAP dependence plots show how a single feature's value affects the SHAP value (and thus the prediction),
        potentially highlighting interactions with another feature.
        **(Note: These are placeholder/example images.)**
    """)
    # Example:
    # dep_plot1_path = f"assets/shap_dependence_plot1_{selected_model_name.lower().replace(' ', '_')}.png"
    # if os.path.exists(dep_plot1_path):
    #     image_dep1 = Image.open(dep_plot1_path)
    #     st.image(image_dep1, caption=f"Example SHAP Dependence Plot 1 for {selected_model_name}")
    # else:
    #     st.info(f"No pre-generated SHAP dependence plot 1 found for {selected_model_name}.")


    st.markdown("---")
    st.subheader("Geological Meaning of Influential Variables (Discussion)")
    st.markdown("""
    *(This section should be filled with your discussion of the 5-10 most influential variables
    and their geological meaning, based on your project's interpretability analysis.)*

    For example:
    * **SiO₂ (Silica):** Higher silica content often correlates with more felsic magmas, which can be associated with certain types of porphyry deposits...
    * **Cu/Au Ratio (if engineered):** This directly informs the classification...
    * **K₂O/Na₂O:** Indicates alkalinity, which plays a role in magma evolution and mineralization...
    * **Sr/Y Ratio:** Can be an indicator of slab melt involvement or crustal thickness...
    """)
