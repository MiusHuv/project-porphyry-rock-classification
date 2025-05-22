# geochem_classifier_gui/pages/3_📈_Performance_Visualizer.py
import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from core.visualizer import plot_confusion_matrix_func, plot_roc_curve_func, plot_precision_recall_curve_func, CLASS_NAMES

def show_page():
    st.title("📈 Model Performance Visualizer")

    if 'predictions_df' not in st.session_state or st.session_state.predictions_df is None:
        st.warning("No predictions found. Please run predictions on the '🚀 Run Prediction' page first.")
        st.stop()

    if 'true_label_column' not in st.session_state or st.session_state.true_label_column is None:
        st.info("""
            To visualize performance, please ensure you uploaded a file with a 'True Label' column
            and selected it on the '🚀 Run Prediction' page.
        """)
        st.stop()

    predictions_df = st.session_state.predictions_df
    true_label_col = st.session_state.true_label_column
    raw_df_with_labels = st.session_state.raw_df_with_labels # Get the original df

    if true_label_col not in raw_df_with_labels.columns:
        st.error(f"The specified true label column '{true_label_col}' was not found in the uploaded data.")
        st.stop()

    # Extract true labels and predicted labels
    # Ensure consistent mapping of true labels to numeric (0 or 1) if they are strings
    # This mapping MUST match your training label encoding and CLASS_NAMES
    # For example, if CLASS_NAMES = ['Au-rich', 'Cu-rich']
    # Then 'Au-rich' -> 0, 'Cu-rich' -> 1

    # Create a mapping from string labels to integers based on CLASS_NAMES
    label_to_int_mapping = {label: i for i, label in enumerate(CLASS_NAMES)}

    try:
        y_true_str = raw_df_with_labels[true_label_col].astype(str) # Ensure string type for mapping
        y_true = y_true_str.map(label_to_int_mapping)
        
        # Check if all true labels were successfully mapped
        if y_true.isnull().any():
            unmapped_labels = y_true_str[y_true.isnull()].unique()
            st.error(f"Some true labels could not be mapped to numeric values: {unmapped_labels}. "
                     f"Ensure labels in '{true_label_col}' are one of {CLASS_NAMES}.")
            st.stop()

    except KeyError: # If a label in y_true_str is not in label_to_int_mapping
        st.error(f"Error mapping true labels. Ensure labels in column '{true_label_col}' are one of {CLASS_NAMES}.")
        st.stop()


    # Predicted labels are already numeric (0 or 1 from predictor.py, then mapped to string in Run_Prediction)
    # We need the numeric predictions again for metrics, or map the string predictions back.
    # Let's re-map the string predictions from results_df.
    y_pred_str = predictions_df['Predicted Class']
    y_pred = y_pred_str.map(label_to_int_mapping)
    
    if y_pred.isnull().any():
        st.error("Error processing predicted classes for performance metrics.")
        st.stop()

    # Probabilities are for the positive class ('Cu-rich', which is CLASS_NAMES[1])
    y_pred_proba = predictions_df['Prediction Probability (Cu-rich)']


    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
    # Specify positive label for precision, recall, F1 if CLASS_NAMES[1] is 'Cu-rich'
    # And label_to_int_mapping[CLASS_NAMES[1]] gives its numeric representation (e.g., 1)
    positive_label_numeric = label_to_int_mapping[CLASS_NAMES[1]] 
    
    col2.metric(f"Precision ({CLASS_NAMES[1]})", f"{precision_score(y_true, y_pred, pos_label=positive_label_numeric, zero_division=0):.3f}")
    col3.metric(f"Recall ({CLASS_NAMES[1]})", f"{recall_score(y_true, y_pred, pos_label=positive_label_numeric, zero_division=0):.3f}")
    col4.metric(f"F1-Score ({CLASS_NAMES[1]})", f"{f1_score(y_true, y_pred, pos_label=positive_label_numeric, zero_division=0):.3f}")

    st.subheader("Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"])

    with tab1:
        st.markdown("#### Confusion Matrix")
        cm_fig = plot_confusion_matrix_func(y_true, y_pred, class_names=CLASS_NAMES)
        st.pyplot(cm_fig)
        st.caption(f"""
            Shows the performance of the classification model.
            Rows represent the actual classes, and columns represent the predicted classes.
            - **{CLASS_NAMES[1]}**: Copper-dominant porphyry samples.
            - **{CLASS_NAMES[0]}**: Gold-dominant porphyry samples.
        """)

    with tab2:
        st.markdown("#### ROC Curve")
        # Model name for plot title - extract from predictions_df if stored, or use generic
        model_name_used = "Selected Model" # Placeholder
        if 'predictions_df' in st.session_state and st.session_state.predictions_df is not None:
             # This assumes you might have saved the model name with predictions, which is good practice
             # For simplicity, I didn't explicitly add it to the predictions_df in Run_Prediction page,
             # but you could add a column like 'Model Used'.
             # For now, we'll just use a generic title or try to get it from current selection if possible.
             if 'selected_model_name' in st.session_state : # If it's still in session from Run Prediction
                 model_name_used = st.session_state.selected_model_name

        roc_fig = plot_roc_curve_func(y_true, y_pred_proba, model_name=model_name_used)
        st.pyplot(roc_fig)
        st.caption("""
            The Receiver Operating Characteristic (ROC) curve illustrates the diagnostic ability of the classifier
            as its discrimination threshold is varied. The Area Under the Curve (AUC) measures the
            entire two-dimensional area underneath the entire ROC curve from (0,0) to (1,1).
            A model with 100% accuracy has an AUC of 1.0.
        """)

    with tab3:
        st.markdown("#### Precision-Recall Curve")
        pr_fig = plot_precision_recall_curve_func(y_true, y_pred_proba, model_name=model_name_used)
        st.pyplot(pr_fig)
        st.caption("""
            The Precision-Recall curve shows the tradeoff between precision and recall for different thresholds.
            A high area under the curve represents both high recall and high precision.
            AP (Average Precision) summarizes this curve.
        """)
