# page/model_insights.py
from pathlib import Path
import streamlit as st
from PIL import Image
import os
import json
import pandas as pd
import numpy as np
from core.model_loader import load_selected_model, load_gui_preprocessor_artifacts
from core.visualizer import get_feature_importances_for_model, plot_feature_importances_func
from util.language import T

def show_page():
    st.title(T("model_insights_title"))
    st.markdown(T("model_insights_intro"))

    gui_artifacts = load_gui_preprocessor_artifacts()
    top_shap_features_config = {}
    top_features_json_path = Path(os.path.dirname(os.path.dirname(__file__))) / "models" / "top_shap_features_for_dependence_plots.json"

    if top_features_json_path.exists():
        try:
            with open(top_features_json_path, 'r') as f:
                top_shap_features_config = json.load(f)

        except Exception as e:
            top_shap_features_config = {}
    else:
        st.error(T("model_insights_top_features_load_error"))

    if gui_artifacts is None or not gui_artifacts.get("final_feature_names"):
        st.error("Could not load necessary artifacts (like feature names) for model insights. Please check model training and saving.")
        st.stop()
    
    FINAL_FEATURE_NAMES_FROM_TRAINING = gui_artifacts["final_feature_names"]
    CLASS_NAMES_FROM_TRAINING = gui_artifacts["label_encoder_classes"]

    model_options = ["Random Forest", "XGBoost", "SVM", "PyTorch DNN"] # Consistent naming
    selected_model_name_insights = st.selectbox(
        T("model_insights_model_select_label"),
        model_options,
        key="insights_model_select_main"
    )

    st.subheader(T("model_insights_subheader_for_model", model_name=selected_model_name_insights))

    # --- Feature Importances ---
    if selected_model_name_insights != "PyTorch DNN": 
        st.markdown(T("model_insights_fi_header"))
        
        fi_image_base_path = Path(os.path.dirname(os.path.dirname(__file__))) / "assets" / "importance_plots"
        model_filename_key = selected_model_name_insights.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
        
        fi_image_path = fi_image_base_path / f"{model_filename_key}_feature_importances.png"


        if fi_image_path.exists():
            try:
                fi_image = Image.open(fi_image_path)
                st.image(fi_image, caption=T("model_insights_fi_plot_caption"))
            except Exception as e:
                st.warning(T("model_insights_fi_image_load_warning", 
                            error_message=str(e)))
                
                model_for_insights = load_selected_model(selected_model_name_insights)
                if model_for_insights and FINAL_FEATURE_NAMES_FROM_TRAINING:
                    importances_series = get_feature_importances_for_model(model_for_insights, FINAL_FEATURE_NAMES_FROM_TRAINING, selected_model_name_insights)
                    if importances_series is not None and not importances_series.empty:
                        fig = plot_feature_importances_func(importances_series, selected_model_name_insights)
                        if fig:
                            st.pyplot(fig)
                            st.caption(T("model_insights_fi_plot_caption"))
                        else:
                            st.info(T("model_insights_fi_plot_error"))
        else:
            # Image not found, show info and try to generate dynamically
            st.info(T("model_insights_fi_image_not_found", 
                    model_name=selected_model_name_insights, 
                    image_path=f"{model_filename_key}_feature_importances.png"))

            # Try to generate dynamically as a fallback
            model_for_insights = load_selected_model(selected_model_name_insights)
            if model_for_insights and FINAL_FEATURE_NAMES_FROM_TRAINING:
                importances_series = get_feature_importances_for_model(model_for_insights, FINAL_FEATURE_NAMES_FROM_TRAINING, selected_model_name_insights)
                if importances_series is not None and not importances_series.empty:
                    fig = plot_feature_importances_func(importances_series, selected_model_name_insights)
                    if fig:
                        st.pyplot(fig)
                        st.caption(T("model_insights_fi_plot_caption"))
                    else:
                        st.info(T("model_insights_fi_plot_error"))
                else:
                    st.info(T("model_insights_fi_unavailable_info", model_name=selected_model_name_insights))
            else:
                st.warning(T("model_insights_fi_load_warning", model_name=selected_model_name_insights))
    else:
        st.info(T("model_insights_fi_dnn_info"))

    # --- SHAP Plots (Static Images) ---
    st.markdown(T("model_insights_shap_summary_header"))
    st.write(T("model_insights_shap_summary_desc"))

    shap_image_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets/shap_plots")
    # Ensure filename consistency with training_pipeline.py SHAP saving
    model_filename_key_shap = selected_model_name_insights.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')

    # SHAP Summary Plot
    
    summary_plot_found = False
    if CLASS_NAMES_FROM_TRAINING and len(CLASS_NAMES_FROM_TRAINING) == 2: # Binary case
        positive_class_label_safe = CLASS_NAMES_FROM_TRAINING[1].lower().replace(' ', '_').replace('-', '_')
        shap_summary_image_filename = f"{model_filename_key_shap}_shap_summary_{positive_class_label_safe}.png"
        # Also check for a non-class-specific one that generate_and_save_shap_plots might create for binary
        generic_binary_shap_summary_filename = f"{model_filename_key_shap}_shap_summary.png"
        
        shap_summary_image_path_class_specific = os.path.join(shap_image_base_path, shap_summary_image_filename)
        shap_summary_image_path_generic = os.path.join(shap_image_base_path, generic_binary_shap_summary_filename)

        if os.path.exists(shap_summary_image_path_class_specific):
            shap_summary_image_path = shap_summary_image_path_class_specific
            summary_plot_found = True
        elif os.path.exists(shap_summary_image_path_generic):
            shap_summary_image_path = shap_summary_image_path_generic
            summary_plot_found = True
    else: # Attempt generic name for multi-class (though per-class is better) or if class names not loaded
        shap_summary_image_filename = f"{model_filename_key_shap}_shap_summary.png"
        shap_summary_image_path = os.path.join(shap_image_base_path, shap_summary_image_filename)
        if os.path.exists(shap_summary_image_path):
            summary_plot_found = True


    if summary_plot_found:
        try:
            image = Image.open(shap_summary_image_path)
            st.image(image, caption=T("model_insights_shap_summary_caption", model_name=selected_model_name_insights))
        except Exception as e:
            st.warning(T("model_insights_shap_image_load_warning", image_type="Summary", error_message=str(e)))
    else:
        st.info(T("model_insights_shap_image_not_found", image_type="Summary", model_name=selected_model_name_insights, image_path=f"{model_filename_key_shap}_shap_summary...png"))


    st.markdown(T("model_insights_shap_dependence_header"))
    st.write(T("model_insights_shap_dependence_desc"))

    # SHAP Dependence Plots
    shap_image_base_path = Path(os.path.dirname(os.path.dirname(__file__))) / "assets" / "shap_plots"

    if FINAL_FEATURE_NAMES_FROM_TRAINING and len(FINAL_FEATURE_NAMES_FROM_TRAINING) >= 1:
        features_for_dep_plots = top_shap_features_config.get(selected_model_name_insights, [])
        st.info(f"Features for dependence plots: {features_for_dep_plots}")
        
        if not features_for_dep_plots:
            st.info(T("model_insights_shap_dependence_config_missing", model_name=selected_model_name_insights))
        else:
            features_to_display = features_for_dep_plots[:min(2, len(features_for_dep_plots))]
            
            for feature_name in features_to_display:
                feature_name_safe = feature_name.lower().replace('/', '_').replace(':', '_').replace('(', '').replace(')', '').replace('.', '')
                
                dep_plot_found = False
                dep_plot_path = None

                potential_paths = [
                    shap_image_base_path / f"{model_filename_key_shap}_shap_dependence_{feature_name_safe}.png",
                ]

                for path in potential_paths:
                    if path.exists():
                        dep_plot_path = path
                        dep_plot_found = True
                        break

                if dep_plot_found:
                    try:
                        image_dep = Image.open(dep_plot_path)
                        st.image(image_dep, caption=T("model_insights_shap_dependence_caption", 
                                                    feature_name=feature_name, 
                                                    model_name=selected_model_name_insights))
                    except Exception as e:
                        st.warning(T("model_insights_shap_image_load_warning", 
                                    image_type=f"Dependence ({feature_name})", 
                                    error_message=str(e)))
                else:
                    st.info(T("model_insights_shap_image_not_found", 
                            image_type=f"Dependence ({feature_name})", 
                            model_name=selected_model_name_insights, 
                            image_path=f"{model_filename_key_shap}_shap_dependence_{feature_name_safe}...png"))
    else:
        st.info(T("model_insights_shap_dependence_config_missing", model_name=selected_model_name_insights))


    st.markdown("---")
    st.subheader(T("model_insights_geological_meaning_header"))
    st.markdown(T("model_insights_geological_meaning_desc")) # User to fill this

if __name__ == "__main__":
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    # Mock artifacts for standalone testing
    # ... (similar dummy file creation as in run_prediction.py if needed for testing paths) ...
    show_page()