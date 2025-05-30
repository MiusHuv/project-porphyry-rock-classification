# page/model_insights.py
import streamlit as st
from PIL import Image
import os
import pandas as pd
import numpy as np
from core.model_loader import load_selected_model, load_gui_preprocessor_artifacts
# We need a function to get feature importances; this might be in visualizer or model_loader
# For now, let's assume it can be adapted or called.
# from core.visualizer import get_feature_importances_for_model, plot_feature_importances_func
from util.language import T

# Placeholder for get_feature_importances_for_model and plot_feature_importances_func
# These would ideally live in core.visualizer or a similar utility module
def get_feature_importances_for_model(model, feature_names, model_type):
    if hasattr(model, 'feature_importances_'): # Tree-based models (RF, XGBoost)
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    elif hasattr(model, 'coef_') and model_type == "SVM" and model.kernel == 'linear': # Linear SVM
        # For multi-class linear SVM, coef_ can be (n_classes, n_features)
        # For binary, it's often (1, n_features). We'll take absolute mean across classes if multi-class for simplicity.
        if model.coef_.ndim == 2 and model.coef_.shape[0] > 1:
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            importances = np.abs(model.coef_).flatten()
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)
    # Add logic for permutation importance for non-linear SVM or other models if needed
    # This is a simplified version. Production might need permutation importance for some models.
    return None

def plot_feature_importances_func(importances_series, model_name, top_n=20):
    if importances_series is None or importances_series.empty:
        return None
    import matplotlib.pyplot as plt
    import seaborn as sns

    top_n_importances = importances_series.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_n_importances) / 2)))
    sns.barplot(x=top_n_importances.values, y=top_n_importances.index, hue=top_n_importances.index, legend=False, ax=ax, palette="viridis")
    ax.set_title(T("viz_fi_title", n_features=top_n, model_name=model_name))
    ax.set_xlabel(T("viz_fi_xlabel"))
    ax.set_ylabel(T("viz_fi_ylabel"))
    plt.tight_layout()
    return fig


def show_page():
    st.title(T("model_insights_title"))
    st.markdown(T("model_insights_intro"))

    gui_artifacts = load_gui_preprocessor_artifacts()
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
    if selected_model_name_insights != "PyTorch DNN": # DNN feature importances are complex
        st.markdown(T("model_insights_fi_header"))
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
    # For multi-class, training_pipeline saves per-class plots. e.g., shap_summary_modelname_classlabel.png
    # For binary, it might save one, or one for the positive class.
    # The current generate_and_save_shap_plots saves per class if shap_values is a list.
    # Let's assume for binary, it saves shap_summary_modelnamekey.png (for positive class or overall)
    # And for multi-class, it might be shap_summary_modelnamekey_classname.png

    # Try a generic summary plot first, then class-specific if binary and CLASS_NAMES available
    generic_summary_filename = f"{model_filename_key_shap}_shap_summary.png" # from generate_and_save_shap_plots
    # If your generate_and_save_shap_plots saves like `shap_summary_modelnamekey_class_0.png` / `_class_1.png`
    # you'll need to adjust this logic or load specific class plots.
    # For simplicity, let's try to load a plot for the positive class (CLASS_NAMES_FROM_TRAINING[1]) if it's binary.
    
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

    # Modified logic: select features based on the chosen model
    if selected_model_name_insights == "Random Forest":
        features_for_dep_plots = ["nb", "ta", "tio2_ppm"]
    elif selected_model_name_insights == "XGBoost":
        features_for_dep_plots = ["MgO_ppm", "Nb", "TiO2_ppm", "TFe2O3_ppm"]
    elif selected_model_name_insights == "SVM":
        features_for_dep_plots = ["Nb", "TFe2O3_ppm", "TiO2_ppm"]
    elif selected_model_name_insights == "PyTorch DNN":
        features_for_dep_plots = ["hf", "mgo_ppm", "nb", "tfe2o3_ppm", "tio2_ppm"]
    else:
        features_for_dep_plots = []  # fallback if needed

    # Loop over the chosen features for SHAP dependence plots
    for feature_name in features_for_dep_plots:
        feature_name_safe = feature_name.replace('/', '_').replace(':', '_').lower() # Matching saving logic

        # Try to load plot for the positive class in binary case or a generic one if not found
        dep_plot_path_to_try = ""
        dep_plot_found = False

        print(f'CLASS_NAMES_FROM_TRAINING的长度为：{len(CLASS_NAMES_FROM_TRAINING)}')
        
        if CLASS_NAMES_FROM_TRAINING and len(CLASS_NAMES_FROM_TRAINING) == 2:
            positive_class_label_safe = CLASS_NAMES_FROM_TRAINING[1].lower().replace(' ', '_').replace('-', '_')
            print(f'positive_class_label_safe的值为：{positive_class_label_safe}')
            # dep_filename_class_specific = f"{model_filename_key_shap}_shap_dependence_{feature_name_safe}_{positive_class_label_safe}.png"
            dep_filename_class_specific = f"{model_filename_key_shap}_shap_dependence_{feature_name_safe}.png"
            path_class_specific = os.path.join(shap_image_base_path, dep_filename_class_specific)
            if os.path.exists(path_class_specific):
                dep_plot_path_to_try = path_class_specific
                dep_plot_found = True
        else:
            print("这个条件判断为false！")

        # if not dep_plot_found:  # Try generic filename as fallback
        #     dep_filename_generic = f"{model_filename_key_shap}_shap_dependence_{feature_name_safe}.png"
        #     # dep_filename_generic = f"{model_filename_key_shap}_shap_dependence_{feature_name_safe}_{positive_class_label_safe}.png"
        #     path_generic = os.path.join(shap_image_base_path, dep_filename_generic)
        #     if os.path.exists(path_generic):
        #         dep_plot_path_to_try = path_generic
        #         dep_plot_found = True

        if dep_plot_found:
            try:
                image_dep = Image.open(dep_plot_path_to_try)
                st.image(image_dep, caption=T("model_insights_shap_dependence_caption", feature_name=feature_name, model_name=selected_model_name_insights))
            except Exception as e:
                st.warning(T("model_insights_shap_image_load_warning", image_type=f"Dependence ({feature_name})", error_message=str(e)))
        else:
            st.info(T("model_insights_shap_image_not_found", image_type=f"Dependence ({feature_name})", model_name=selected_model_name_insights, image_path=f"{model_filename_key_shap}_shap_dependence_{feature_name_safe}...png"))
    # else:
    #     st.info(T("model_insights_shap_dependence_config_missing", model_name=selected_model_name_insights))


    st.markdown("---")
    st.subheader(T("model_insights_geological_meaning_header"))
    st.markdown(T("model_insights_geological_meaning_desc")) # User to fill this

if __name__ == "__main__":
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    # Mock artifacts for standalone testing
    # ... (similar dummy file creation as in run_prediction.py if needed for testing paths) ...
    show_page()