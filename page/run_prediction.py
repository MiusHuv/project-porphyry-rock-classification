# geochem_classifier_gui/pages/2_🚀_Run_Prediction.py
import streamlit as st
import pandas as pd
from core.data_handler import load_data, validate_data, preprocess_data_for_prediction, EXPECTED_COLUMNS
from core.model_loader import load_selected_model, load_preprocessor_object
from core.predictor import make_predictions
from core.visualizer import CLASS_NAMES # To map numeric predictions back to labels
from util.language import T, TEXTS

def show_page():
    st.title(T("run_pred_title")) # "Run Prediction on New Samples" -> "Run Prediction on New Data"

    # --- File Upload and Model Selection (Sidebar) ---
    st.sidebar.header(T("controls_header", default="Controls")) # New key
    uploaded_file = st.sidebar.file_uploader(
        T("run_pred_upload_label_sidebar"), # "Upload your sample data (.csv or .xlsx)" -> "Upload an Excel (.xlsx) or CSV (.csv) file:"
        type=['csv', 'xlsx'],
        key="run_pred_file_uploader" # Added key
    )
    
    model_options = ["Random Forest", "XGBoost", "SVM", "DNN-Keras"] # These could be keys if model names need translation
    selected_model_name = st.sidebar.selectbox(
        T("run_pred_model_label_sidebar"), # "Choose a classification model:" -> "Choose a pre-trained model:"
        model_options,
        key="run_pred_model_selector" # Added key
    )

    # --- Load Preprocessor ---
    @st.cache_resource 
    def get_preprocessor_cached(): # Renamed for clarity
        return load_preprocessor_object()
    
    preprocessor = get_preprocessor_cached()
    if preprocessor is None:
        st.error(T("run_pred_error_preprocessor_load")) # "Critical error: Preprocessor could not be loaded. Predictions cannot proceed."
        st.stop()

    # --- Load Model ---
    @st.cache_resource
    def get_model_cached(model_name_cache_key): # Renamed for clarity and added cache key param
        return load_selected_model(model_name_cache_key)

    model = get_model_cached(selected_model_name) # Use selected_model_name for caching
    if model is None:
        # Using f-string for model name injection, T() handles the base string
        st.sidebar.error(T("run_pred_error_model_load_sidebar", model_name=selected_model_name, default=f"Model {selected_model_name} could not be loaded.")) # New key
        st.stop()
    else:
        st.sidebar.success(T("run_pred_success_model_ready_sidebar", model_name=selected_model_name, default=f"{selected_model_name} model ready.")) # New key

    # --- Data Handling & Prediction (Main Page) ---
    if uploaded_file is None:
        st.info(T("run_pred_awaiting_upload", default="Awaiting data file upload in the sidebar...")) # New key
        st.stop() # Stop if no file is uploaded

    raw_df = load_data(uploaded_file)

    if raw_df is None:
        # load_data should ideally show its own error, but as a fallback:
        st.error(T("run_pred_error_loading_data", default="Failed to load data from the uploaded file.")) # New key
        st.stop()

    st.subheader(T("run_pred_input_data_header")) # "1. Uploaded Data Preview (First 5 Rows)"
    st.dataframe(raw_df.head())

    # Allow user to specify true label column if present for later evaluation
    true_label_col_name = None
    # Reset session state for these if file changes or checkbox is untoggled
    st.session_state.true_label_column = None
    st.session_state.raw_df_with_labels = None

    if st.checkbox(T("run_pred_true_label_checkbox"), key="true_label_checkbox_runpred"): # "My data includes a 'True Label' column for performance evaluation"
        potential_label_cols = [col for col in raw_df.columns if col not in EXPECTED_COLUMNS]
        if not potential_label_cols:
            potential_label_cols = list(raw_df.columns) 
        
        if potential_label_cols:
            true_label_col_name = st.selectbox(
                T("run_pred_true_label_select"), # "Select your 'True Label' column:" -> "Select the column containing true labels:"
                options=potential_label_cols, 
                index=0,
                help=T("run_pred_true_label_help", default="This column will be used for performance metrics if available."), # New key
                key="true_label_selector_runpred" # Added key
            )
            st.session_state.true_label_column = true_label_col_name 
            st.session_state.raw_df_with_labels = raw_df.copy() # Store the full df with a copy
        else:
            st.warning(T("run_pred_warning_no_label_col", default="Could not identify a potential label column.")) # New key

    st.subheader(T("run_pred_validation_header", default="2. Data Validation & Preprocessing")) # New key
    feature_df, validation_msg = validate_data(raw_df.copy()) 
    
    if feature_df is None:
        st.error(T("run_pred_error_validation_failed", message=validation_msg, default=f"Validation failed: {validation_msg}")) # New key
    else:
        st.success(T("run_pred_success_validation", message=validation_msg, default=validation_msg)) # New key
        
        with st.spinner(T("run_pred_spinner_preprocessing", default="Preprocessing data...")): # New key
            processed_df_for_prediction = preprocess_data_for_prediction(feature_df, preprocessor)

        if processed_df_for_prediction is not None:
            st.write(T("run_pred_success_preprocessing_done", default="Data preprocessed successfully for prediction.")) # New key
            # with st.expander(T("run_pred_expander_processed_data", default="View Processed Data (First 5 Rows)")): # New key
            #     st.dataframe(processed_df_for_prediction.head())

            st.subheader(T("run_pred_run_prediction_header", default="3. Run Prediction")) # New key
            # Using f-string for model name, T() for base string
            if st.button(T("run_pred_submit_button_main", model_name=selected_model_name, default=f"Predict using {selected_model_name}"), type="primary", key="run_predict_button_main"): # New key
                with st.spinner(T("run_pred_processing_spinner")): # "Running predictions..."
                    try:
                        predictions_numeric, probabilities = make_predictions(model, processed_df_for_prediction, selected_model_name)
                        
                        if predictions_numeric is not None and probabilities is not None:
                            results_df = raw_df.copy()
                            
                            # Use T() for column names if they need to be dynamic/translated
                            predicted_class_col_key = "results_col_predicted_class"
                            predicted_class_col_default = 'Predicted Class'
                            prob_col_name_key = "results_col_probability_cu_rich"
                            prob_col_name_default = f'Prediction Probability ({CLASS_NAMES[1]})'


                            results_df[T(predicted_class_col_key, default=predicted_class_col_default)] = [CLASS_NAMES[p] for p in predictions_numeric]
                            results_df[T(prob_col_name_key, default=prob_col_name_default)] = probabilities 

                            st.session_state.predictions_df = results_df

                            st.subheader(T("run_pred_results_header")) # "4. Prediction Results"
                            st.dataframe(results_df)

                            cu_rich_count = (results_df[T(predicted_class_col_key, default=predicted_class_col_default)] == CLASS_NAMES[1]).sum()
                            au_rich_count = (results_df[T(predicted_class_col_key, default=predicted_class_col_default)] == CLASS_NAMES[0]).sum()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(label=T("run_pred_summary_cu_label", default=f"Predicted {CLASS_NAMES[1]} Samples"), value=cu_rich_count) # New key
                            with col2:
                                st.metric(label=T("run_pred_summary_au_label", default=f"Predicted {CLASS_NAMES[0]} Samples"), value=au_rich_count) # New key


                            csv_data = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=T("run_pred_download_button"), # "Download Predictions as CSV"
                                data=csv_data,
                                file_name=f"predictions_{selected_model_name.replace(' ', '_').lower()}.csv",
                                mime='text/csv',
                                key="download_predictions_csv_runpred" # Added key
                            )
                            st.success(T("run_pred_success_prediction_done", default="Predictions complete! Results are shown above and can be downloaded.")) # New key
                        else:
                            st.error(T("run_pred_error_prediction_failed_internal", default="Prediction process failed internally. Model might not be compatible or data issue.")) # New key
                    except RuntimeError as e:
                        st.error(T("run_pred_error_runtime_prediction", error_message=str(e), default=f"An error occurred during prediction: {e}")) # New key
                    except Exception as e:
                        st.error(T("run_pred_error_unexpected", error_message=str(e), default=f"An unexpected error occurred: {e}")) # New key
        else:
            st.error(T("run_pred_error_preprocessing_failed", default="Data preprocessing failed. Check data compatibility.")) # New key


if __name__ == "__main__":
    # Initialize session state variables if not already present
    if 'lang' not in st.session_state: st.session_state.lang = "en" # For T() to work
    if 'predictions_df' not in st.session_state:
        st.session_state.predictions_df = None
    if 'true_label_column' not in st.session_state:
        st.session_state.true_label_column = None
    if 'raw_df_with_labels' not in st.session_state:
        st.session_state.raw_df_with_labels = None
    show_page()