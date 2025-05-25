# ./page/Performance_Visualizer.py

import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from core.visualizer import plot_confusion_matrix_func, plot_roc_curve_func, plot_precision_recall_curve_func, CLASS_NAMES
from util.language import T

def show_page():
    st.title(T("perf_viz_title", default="Model Performance Visualizer"))

    if 'predictions_df' not in st.session_state or st.session_state.predictions_df is None:
        st.warning(T("perf_viz_no_predictions_warning", default="No predictions found. Please run predictions on the 'Run Prediction' page first."))
        st.stop()

    if 'true_label_column' not in st.session_state or st.session_state.true_label_column is None:
        st.info(T("perf_viz_no_true_label_info", default="""
            To visualize performance, please ensure you uploaded a file with a 'True Label' column
            and selected it on the 'Run Prediction' page.
        """))
        st.stop()

    predictions_df = st.session_state.predictions_df
    true_label_col = st.session_state.true_label_column
    
    # Ensure raw_df_with_labels is available from session state
    if 'raw_df_with_labels' not in st.session_state or st.session_state.raw_df_with_labels is None:
        st.error(T("perf_viz_raw_data_missing_error", default="Original data with labels is missing from session. Please re-run prediction with true labels specified."))
        st.stop()
    raw_df_with_labels = st.session_state.raw_df_with_labels

    if true_label_col not in raw_df_with_labels.columns:
        st.error(T("perf_viz_true_label_not_found_error", true_label_col=true_label_col, default=f"The specified true label column '{true_label_col}' was not found in the uploaded data."))
        st.stop()

    label_to_int_mapping = {label: i for i, label in enumerate(CLASS_NAMES)}
    # Ensure CLASS_NAMES has at least two elements for typical binary classification
    if len(CLASS_NAMES) < 2:
        st.error(T("perf_viz_class_names_error", default="CLASS_NAMES configuration is invalid for binary classification visualization."))
        st.stop()

    try:
        y_true_str = raw_df_with_labels[true_label_col].astype(str)
        y_true = y_true_str.map(label_to_int_mapping)
        
        if y_true.isnull().any():
            unmapped_labels = y_true_str[y_true.isnull()].unique()
            st.error(T("perf_viz_label_mapping_error", unmapped_labels=str(unmapped_labels), true_label_col=true_label_col, class_names_list=str(CLASS_NAMES), default=f"Some true labels ({unmapped_labels}) in column '{true_label_col}' could not be mapped. Ensure labels are one of {CLASS_NAMES}."))
            st.stop()
    except KeyError:
        st.error(T("perf_viz_true_label_mapping_key_error", true_label_col=true_label_col, class_names_list=str(CLASS_NAMES), default=f"Error mapping true labels. Ensure labels in column '{true_label_col}' are one of {CLASS_NAMES}."))
        st.stop()

    # Get translated column names from run_prediction.py
    predicted_class_col_name = T("results_col_predicted_class", default='Predicted Class')
    prob_col_name = T("results_col_probability_cu_rich", class_name_cu_rich=CLASS_NAMES[1], default=f'Prediction Probability ({CLASS_NAMES[1]})')

    if predicted_class_col_name not in predictions_df.columns:
        st.error(T("perf_viz_predicted_class_col_missing", col_name=predicted_class_col_name, default=f"'{predicted_class_col_name}' column not found in prediction results."))
        st.stop()
    if prob_col_name not in predictions_df.columns:
        st.error(T("perf_viz_probability_col_missing", col_name=prob_col_name, default=f"'{prob_col_name}' column not found in prediction results."))
        st.stop()

    y_pred_str = predictions_df[predicted_class_col_name]
    y_pred = y_pred_str.map(label_to_int_mapping)
    
    if y_pred.isnull().any():
        st.error(T("perf_viz_pred_label_processing_error", default="Error processing predicted classes for performance metrics."))
        st.stop()

    y_pred_proba = predictions_df[prob_col_name]

    st.subheader(T("perf_viz_metrics_subheader", default="Performance Metrics"))
    
    col1, col2, col3, col4 = st.columns(4)
    positive_label_numeric = label_to_int_mapping[CLASS_NAMES[1]] 
    
    col1.metric(T("perf_viz_accuracy_label", default="Accuracy"), f"{accuracy_score(y_true, y_pred):.3f}")
    col2.metric(T("perf_viz_precision_label", class_name=CLASS_NAMES[1], default=f"Precision ({CLASS_NAMES[1]})"), f"{precision_score(y_true, y_pred, pos_label=positive_label_numeric, zero_division=0):.3f}")
    col3.metric(T("perf_viz_recall_label", class_name=CLASS_NAMES[1], default=f"Recall ({CLASS_NAMES[1]})"), f"{recall_score(y_true, y_pred, pos_label=positive_label_numeric, zero_division=0):.3f}")
    col4.metric(T("perf_viz_f1_label", class_name=CLASS_NAMES[1], default=f"F1-Score ({CLASS_NAMES[1]})"), f"{f1_score(y_true, y_pred, pos_label=positive_label_numeric, zero_division=0):.3f}")

    st.subheader(T("perf_viz_visualizations_subheader", default="Visualizations"))
    
    tab1_title = T("perf_viz_cm_tab", default="Confusion Matrix")
    tab2_title = T("perf_viz_roc_tab", default="ROC Curve")
    tab3_title = T("perf_viz_pr_tab", default="Precision-Recall Curve")
    
    tab1, tab2, tab3 = st.tabs([tab1_title, tab2_title, tab3_title])

    with tab1:
        st.markdown(T("perf_viz_cm_title_markdown", default="#### Confusion Matrix"))
        cm_fig = plot_confusion_matrix_func(y_true, y_pred, class_names=CLASS_NAMES)
        st.pyplot(cm_fig)
        st.caption(T("perf_viz_cm_caption", class_name_1=CLASS_NAMES[1], class_name_0=CLASS_NAMES[0], default=f"""
            Shows the performance of the classification model.
            Rows represent the actual classes, and columns represent the predicted classes.
            - **{CLASS_NAMES[1]}**: Copper-dominant porphyry samples.
            - **{CLASS_NAMES[0]}**: Gold-dominant porphyry samples.
        """))

    with tab2:
        st.markdown(T("perf_viz_roc_title_markdown", default="#### ROC Curve"))
        model_name_used = T("perf_viz_selected_model_placeholder", default="Selected Model") 
        if 'selected_model_name' in st.session_state and st.session_state.selected_model_name:
             model_name_used = st.session_state.selected_model_name

        roc_fig = plot_roc_curve_func(y_true, y_pred_proba, model_name=model_name_used)
        st.pyplot(roc_fig)
        st.caption(T("perf_viz_roc_caption", default="""
            The Receiver Operating Characteristic (ROC) curve illustrates the diagnostic ability of the classifier
            as its discrimination threshold is varied. The Area Under the Curve (AUC) measures the
            entire two-dimensional area underneath the entire ROC curve from (0,0) to (1,1).
            A model with 100% accuracy has an AUC of 1.0.
        """))

    with tab3:
        st.markdown(T("perf_viz_pr_title_markdown", default="#### Precision-Recall Curve"))
        pr_fig = plot_precision_recall_curve_func(y_true, y_pred_proba, model_name=model_name_used)
        st.pyplot(pr_fig)
        st.caption(T("perf_viz_pr_caption", default="""
            The Precision-Recall curve shows the tradeoff between precision and recall for different thresholds.
            A high area under the curve represents both high recall and high precision.
            AP (Average Precision) summarizes this curve.
        """))

if __name__ == "__main__":
    # Mock session state for standalone testing
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    
    # Example data for testing
    CLASS_NAMES = ['Au-rich', 'Cu-rich'] # Ensure this matches your actual CLASS_NAMES
    label_to_int_mapping = {label: i for i, label in enumerate(CLASS_NAMES)}

    # Mock session state data
    st.session_state.predictions_df = pd.DataFrame({
        T("results_col_predicted_class", default='Predicted Class'): [CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[1]],
        T("results_col_probability_cu_rich", class_name_cu_rich=CLASS_NAMES[1], default=f'Prediction Probability ({CLASS_NAMES[1]})'): [0.2, 0.8, 0.3, 0.9, 0.7]
    })
    st.session_state.true_label_column = 'TrueLabel'
    st.session_state.raw_df_with_labels = pd.DataFrame({
        'Feature1': [1,2,3,4,5],
        'TrueLabel': [CLASS_NAMES[0], CLASS_NAMES[0], CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[1]]
    })
    st.session_state.selected_model_name = "Test Model"

    show_page()