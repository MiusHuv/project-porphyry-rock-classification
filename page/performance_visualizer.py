# page/performance_visualizer.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# Use the specific plotting functions from your visualizer if they are defined there
# Assuming plot_confusion_matrix_func, plot_roc_curve_func, plot_precision_recall_curve_func
# are designed to be called with y_true, y_pred/y_proba, and class_names.
from core.visualizer import plot_confusion_matrix_heatmap as plot_confusion_matrix_func # Example alias
from core.visualizer import plot_roc_curves as plot_roc_curve_func # Adapting
from core.visualizer import plot_precision_recall_curves as plot_precision_recall_curve_func # Adapting

from util.language import T
from core.model_loader import load_gui_preprocessor_artifacts # To get class names

def show_page():
    st.title(T("perf_viz_title"))

    gui_artifacts = load_gui_preprocessor_artifacts()
    if gui_artifacts is None or not gui_artifacts.get("label_encoder_classes"):
        st.error("Critical error: Class names could not be loaded. Performance visualization cannot proceed.")
        st.stop()
    CLASS_NAMES = gui_artifacts["label_encoder_classes"] # Use loaded class names

    # Check for prediction results from run_prediction page
    if 'predictions_df_for_perf_viz' not in st.session_state or st.session_state.predictions_df_for_perf_viz is None:
        st.warning(T("perf_viz_no_predictions_warning"))
        st.stop()

    # Check if true labels were provided during prediction
    if 'true_label_column_for_perf_viz' not in st.session_state or st.session_state.true_label_column_for_perf_viz is None:
        st.info(T("perf_viz_no_true_label_info"))
        st.stop()

    predictions_df = st.session_state.predictions_df_for_perf_viz
    true_label_col_name = st.session_state.true_label_column_for_perf_viz
    raw_df_with_labels = st.session_state.raw_df_with_labels_for_perf_viz # This should have been raw_df_full

    if raw_df_with_labels is None or true_label_col_name not in raw_df_with_labels.columns:
        st.error(T("perf_viz_true_label_not_found_error", true_label_col=str(true_label_col_name)))
        st.stop()

    # --- Prepare y_true and y_pred ---
    # CLASS_NAMES[0] is 'Au-rich PCDs', CLASS_NAMES[1] is 'Cu-rich PCDs' (example from dummy data)
    # Label mapping should align with how label_encoder was fit in training
    # Typically, first class in label_encoder.classes_ is 0, second is 1, etc.
    label_to_int_mapping = {label: i for i, label in enumerate(CLASS_NAMES)}
    int_to_label_mapping = {i: label for i, label in enumerate(CLASS_NAMES)}

    try:
        y_true_str = raw_df_with_labels[true_label_col_name].astype(str)
        y_true_numeric = y_true_str.map(label_to_int_mapping)
        if y_true_numeric.isnull().any():
            unmapped = y_true_str[y_true_numeric.isnull()].unique()
            st.error(T("perf_viz_label_mapping_error", unmapped_labels=str(unmapped), true_label_col=true_label_col_name, class_names_list=str(CLASS_NAMES)))
            st.stop()
    except KeyError:
        st.error(T("perf_viz_true_label_mapping_key_error", true_label_col=true_label_col_name, class_names_list=str(CLASS_NAMES)))
        st.stop()

    # Get predicted class and probability columns from the predictions_df
    # The column names in predictions_df were set using T() in run_prediction.py
    predicted_class_col_localized_name = T("results_col_predicted_class")
    # For probability, it was T("results_col_probability_positive", positive_class_name=CLASS_NAMES[1])
    # We need to reconstruct this name or use a fixed key from session state if stored.
    # Assuming CLASS_NAMES[1] is the positive class consistently.
    prob_col_localized_name = T("results_col_probability_positive", positive_class_name=CLASS_NAMES[1])


    if predicted_class_col_localized_name not in predictions_df.columns:
        st.error(T("perf_viz_predicted_class_col_missing", col_name=predicted_class_col_localized_name))
        st.stop()
    if prob_col_localized_name not in predictions_df.columns:
        st.error(T("perf_viz_probability_col_missing", col_name=prob_col_localized_name))
        st.stop()

    y_pred_str = predictions_df[predicted_class_col_localized_name]
    y_pred_numeric = y_pred_str.map(label_to_int_mapping)
    if y_pred_numeric.isnull().any():
        st.error(T("perf_viz_pred_label_processing_error"))
        st.stop()

    y_pred_proba_positive_class = predictions_df[prob_col_localized_name]

    # --- Display Metrics ---
    st.subheader(T("perf_viz_metrics_subheader"))
    col1, col2, col3, col4 = st.columns(4)
    positive_label_numeric_value = label_to_int_mapping[CLASS_NAMES[1]] # e.g., 1 if 'Cu-rich PCDs' is class 1

    col1.metric(T("perf_viz_accuracy_label"), f"{accuracy_score(y_true_numeric, y_pred_numeric):.3f}")
    col2.metric(T("perf_viz_precision_label", class_name=CLASS_NAMES[1]), f"{precision_score(y_true_numeric, y_pred_numeric, pos_label=positive_label_numeric_value, zero_division=0):.3f}")
    col3.metric(T("perf_viz_recall_label", class_name=CLASS_NAMES[1]), f"{recall_score(y_true_numeric, y_pred_numeric, pos_label=positive_label_numeric_value, zero_division=0):.3f}")
    col4.metric(T("perf_viz_f1_label", class_name=CLASS_NAMES[1]), f"{f1_score(y_true_numeric, y_pred_numeric, pos_label=positive_label_numeric_value, zero_division=0):.3f}")

    # --- Display Visualizations ---
    st.subheader(T("perf_viz_visualizations_subheader"))
    tab1, tab2, tab3 = st.tabs([
        T("perf_viz_cm_tab"),
        T("perf_viz_roc_tab"),
        T("perf_viz_pr_tab")
    ])

    model_name_for_plot = st.session_state.get("selected_model_name", "Selected Model") # From run_prediction sidebar

    with tab1: # Confusion Matrix
        st.markdown(T("perf_viz_cm_title_markdown"))
        cm_fig = plot_confusion_matrix_func(y_true_numeric, y_pred_numeric, class_names_list=CLASS_NAMES, model_name=model_name_for_plot) # Pass string names
        if cm_fig: st.pyplot(cm_fig)
        st.caption(T("perf_viz_cm_caption", class_name_1=CLASS_NAMES[1], class_name_0=CLASS_NAMES[0]))

    # For ROC and PR, ensure y_pred_proba_positive_class is suitable.
    # plot_roc_curves and plot_precision_recall_curves from visualizer expect a dictionary of probas.
    # Here we have probas for one model.
    
    # Convert y_pred_proba_positive_class to the format expected by plot_roc_curve_func/plot_precision_recall_curve_func
    # They might expect probabilities for all classes, or just the positive class for binary.
    # The versions in your visualizer.py (plot_roc_curves, plot_precision_recall_curves)
    # take y_pred_probas_dict. We need to adapt or make specific functions.
    
    # Let's assume we adapt to single model prediction:
    # Create a dummy y_pred_probas_dict for a single model
    # If CLASS_NAMES[1] is positive, its probabilities are in y_pred_proba_positive_class
    # Probabilities for CLASS_NAMES[0] would be 1 - y_pred_proba_positive_class
    
    num_samples = len(y_pred_proba_positive_class)
    y_probas_for_plot = np.zeros((num_samples, len(CLASS_NAMES)))
    y_probas_for_plot[:, label_to_int_mapping[CLASS_NAMES[1]]] = y_pred_proba_positive_class
    y_probas_for_plot[:, label_to_int_mapping[CLASS_NAMES[0]]] = 1 - y_pred_proba_positive_class
    
    single_model_probas_dict = {model_name_for_plot: y_probas_for_plot}

    with tab2: # ROC Curve
        st.markdown(T("perf_viz_roc_title_markdown"))
        roc_fig = plot_roc_curve_func(y_true_numeric, single_model_probas_dict, CLASS_NAMES, num_classes=len(CLASS_NAMES))
        if roc_fig: st.pyplot(roc_fig)
        st.caption(T("perf_viz_roc_caption"))

    with tab3: # Precision-Recall Curve
        st.markdown(T("perf_viz_pr_title_markdown"))
        pr_fig = plot_precision_recall_curve_func(y_true_numeric, single_model_probas_dict, CLASS_NAMES, num_classes=len(CLASS_NAMES))
        if pr_fig: st.pyplot(pr_fig)
        st.caption(T("perf_viz_pr_caption"))

if __name__ == "__main__":
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    # Mock for testing
    # ... (you'd need to mock gui_artifacts and session_state for predictions_df_for_perf_viz etc.) ...
    show_page()