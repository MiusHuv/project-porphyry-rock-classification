# page/run_prediction.py
import streamlit as st
import pandas as pd
from core.data_handler import (
    RAW_EXPECTED_GEOCHEMICAL_FEATURES, # Use this for initial column check
    preprocess_data_for_prediction,
    MAJOR_OXIDES_WT_PERCENT, # Needed for preprocess_data_for_prediction
    TRACE_ELEMENTS_PPM       # Needed for preprocess_data_for_prediction
)
from core.model_loader import load_selected_model, load_gui_preprocessor_artifacts
from core.predictor import make_predictions
from util.language import T

import joblib
import json
import os
# --- Helper to load data (can be moved to data_handler if preferred) ---
def load_data_from_file(uploaded_file_obj):
    if uploaded_file_obj is None:
        return None
    try:
        if uploaded_file_obj.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file_obj)
        elif uploaded_file_obj.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file_obj)
        else:
            st.error(T("data_handler_unsupported_file_type"))
            return None
        return df
    except Exception as e:
        st.error(T("data_handler_error_loading_data", error_message=str(e)))
        return None

def validate_input_data(df, expected_features):
    """Validates if all expected raw geochemical features are present."""
    if df is None:
        return None, "No data loaded."
    
    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        msg = T("data_handler_validation_missing_cols", missing_cols_list=", ".join(missing_cols))
        return None, msg
    
    # Select only the expected geochemical features for further processing
    # Any other columns (like potential true labels) will be handled separately
    feature_df = df[expected_features].copy()

    # Check for non-numeric data in feature columns
    for col in feature_df.columns:
        if not pd.api.types.is_numeric_dtype(feature_df[col]):
            try:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                if feature_df[col].isnull().any(): # Coercion introduced NaNs
                    # This warning can be enhanced, for now, SimpleImputer will handle it.
                    st.warning(f"Column '{col}' had non-numeric values coerced to NaN. Imputation will be applied.")
            except Exception as e: # Should not happen if errors='coerce'
                 msg = T("data_handler_validation_non_numeric", column_name=col)
                 return None, msg # Halt if a feature column cannot be made numeric

    msg = T("data_handler_validation_success") # "Data validation successful. Features ready for preprocessing."
    return feature_df, msg


def show_page():
    st.title(T("run_pred_title"))

    # --- Load Preprocessing Artifacts ---
    # This is cached globally for the session.
    gui_artifacts = load_gui_preprocessor_artifacts()
    if gui_artifacts is None or not all(gui_artifacts.values()):
        st.error(T("run_pred_error_preprocessor_load", default="Critical error: Essential preprocessing artifacts (scaler, class names, feature names) could not be loaded. Predictions cannot proceed."))
        st.stop()

    scaler = gui_artifacts["scaler"]
    CLASS_NAMES_FROM_TRAINING = gui_artifacts["label_encoder_classes"] # This is now the source of truth
    FINAL_FEATURE_NAMES_FROM_TRAINING = gui_artifacts["final_feature_names"] # Names after CLR etc.

    if not CLASS_NAMES_FROM_TRAINING or len(CLASS_NAMES_FROM_TRAINING) < 2:
        st.error("Error: Class names could not be loaded or are invalid. Cannot proceed.")
        st.stop()
    if not FINAL_FEATURE_NAMES_FROM_TRAINING:
        st.error("Error: Trained model feature names could not be loaded. Cannot proceed.")
        st.stop()


    # --- File Upload and Model Selection (Sidebar) ---
    st.sidebar.header(T("controls_header", default="Controls"))
    uploaded_file = st.sidebar.file_uploader(
        T("run_pred_upload_label_sidebar"),
        type=['csv', 'xlsx'],
        key="run_pred_file_uploader_main" # Ensure unique key
    )

    model_options = ["Random Forest", "XGBoost", "SVM", "PyTorch DNN"] # Updated DNN name
    selected_model_name = st.sidebar.selectbox(
        T("run_pred_model_label_sidebar"),
        model_options,
        key="run_pred_model_selector_main" # Ensure unique key
    )

    # --- Load Selected Model ---
    # Model loading is cached based on model_name
    model = load_selected_model(selected_model_name)
    if model is None:
        st.sidebar.error(T("run_pred_error_model_load_sidebar", model_name=selected_model_name))
        st.stop()
    else:
        st.sidebar.success(T("run_pred_success_model_ready_sidebar", model_name=selected_model_name))

    # --- Data Handling & Prediction (Main Page) ---
    if uploaded_file is None:
        st.info(T("run_pred_awaiting_upload"))
        st.stop()

    raw_df_full = load_data_from_file(uploaded_file)
    if raw_df_full is None:
        # Error already shown by load_data_from_file
        st.stop()

    st.subheader(T("run_pred_input_data_header"))
    st.dataframe(raw_df_full.head())

    # True label column selection
    st.session_state.true_label_column_for_perf_viz = None
    st.session_state.raw_df_with_labels_for_perf_viz = None

    if st.checkbox(T("run_pred_true_label_checkbox"), key="true_label_checkbox_run_pred_main"):
        # Potential label columns are those NOT in RAW_EXPECTED_GEOCHEMICAL_FEATURES
        potential_label_cols = [col for col in raw_df_full.columns if col not in RAW_EXPECTED_GEOCHEMICAL_FEATURES]
        if not potential_label_cols: # If all columns are features, offer all as choice
            potential_label_cols = list(raw_df_full.columns)

        if potential_label_cols:
            true_label_col_name_selected = st.selectbox(
                T("run_pred_true_label_select"),
                options=potential_label_cols,
                index=0,
                help=T("run_pred_true_label_help"),
                key="true_label_selector_run_pred_main"
            )
            st.session_state.true_label_column_for_perf_viz = true_label_col_name_selected
            st.session_state.raw_df_with_labels_for_perf_viz = raw_df_full.copy() # Store for perf viz
        else:
            st.warning(T("run_pred_warning_no_label_col"))

    st.subheader(T("run_pred_validation_header"))
    # Validate only the expected raw geochemical features
    raw_feature_df, validation_msg = validate_input_data(raw_df_full, RAW_EXPECTED_GEOCHEMICAL_FEATURES)

    if raw_feature_df is None:
        st.error(validation_msg) # Display detailed validation message
        st.stop()
    else:
        st.success(validation_msg)

        with st.spinner(T("run_pred_spinner_preprocessing")):
            try:
                # Preprocess data using the loaded scaler and knowledge of training transformations
                # cols_for_log_transform_config should be passed if log transforms were part of training
                # For this example, assuming no extra log transforms beyond CLR's implicit log.
                # If specific columns were log-transformed in training_pipeline AFTER clr, list them here.
                cols_for_log_config_pred = None # Example: ['some_other_feature_if_logged_in_training']

                processed_df_for_prediction = preprocess_data_for_prediction(
                    raw_feature_df, # Contains only the RAW_EXPECTED_GEOCHEMICAL_FEATURES
                    scaler,         # The loaded scaler object
                    FINAL_FEATURE_NAMES_FROM_TRAINING, # Target feature names for the scaled data
                    MAJOR_OXIDES_WT_PERCENT,
                    TRACE_ELEMENTS_PPM,
                    cols_for_log_transform_config=cols_for_log_config_pred
                )
            except Exception as e:
                st.error(f"Error during data preprocessing for prediction: {e}")
                # import traceback
                # st.error(traceback.format_exc()) # For more detailed debugging if needed
                processed_df_for_prediction = None


        if processed_df_for_prediction is not None:
            st.write(T("run_pred_success_preprocessing_done"))
            # Optional: view processed data
            # with st.expander(T("run_pred_expander_processed_data")):
            #     st.dataframe(processed_df_for_prediction.head())

            st.subheader(T("run_pred_run_prediction_header"))
            if st.button(T("run_pred_submit_button_main", model_name=selected_model_name), type="primary", key="run_predict_button_main_page"):
                with st.spinner(T("run_pred_processing_spinner")):
                    try:
                        # Determine num_classes for DNN
                        num_classes_dnn = len(CLASS_NAMES_FROM_TRAINING)

                        predictions_numeric, probabilities_positive_class = make_predictions(
                            model,
                            processed_df_for_prediction,
                            selected_model_name,
                            num_classes_for_dnn=num_classes_dnn
                        )

                        if predictions_numeric is not None and probabilities_positive_class is not None:
                            # Create results_df starting from the original raw_df_full to keep all original columns
                            results_df = raw_df_full.copy()

                            # Map numeric predictions back to string labels using CLASS_NAMES_FROM_TRAINING
                            # Ensure CLASS_NAMES_FROM_TRAINING[0] is Au-rich, CLASS_NAMES_FROM_TRAINING[1] is Cu-rich
                            # based on your label_encoder from training.
                            predicted_class_labels = [CLASS_NAMES_FROM_TRAINING[p] for p in predictions_numeric]

                            # Define column names for results table using T() for localization
                            pred_col_t_key = "results_col_predicted_class"
                            prob_col_t_key = "results_col_probability_positive" # Generic name first
                            
                            # The second class from label_encoder.classes_ is assumed to be the "positive" one (e.g., "Cu-rich PCDs")
                            positive_class_name_for_prob_col = CLASS_NAMES_FROM_TRAINING[1]

                            results_df[T(pred_col_t_key)] = predicted_class_labels
                            results_df[T(prob_col_t_key, positive_class_name=positive_class_name_for_prob_col)] = probabilities_positive_class

                            st.session_state.predictions_df_for_perf_viz = results_df # Store for performance viz

                            st.subheader(T("run_pred_results_header"))
                            st.dataframe(results_df)

                            # Summary counts
                            positive_class_count = sum(p == 1 for p in predictions_numeric) # Count of class index 1
                            negative_class_count = sum(p == 0 for p in predictions_numeric) # Count of class index 0

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(label=T("run_pred_summary_positive_label", class_name=CLASS_NAMES_FROM_TRAINING[1]), value=positive_class_count)
                            with col2:
                                st.metric(label=T("run_pred_summary_negative_label", class_name=CLASS_NAMES_FROM_TRAINING[0]), value=negative_class_count)

                            csv_data = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label=T("run_pred_download_button"),
                                data=csv_data,
                                file_name=f"predictions_{selected_model_name.lower().replace(' ', '_').replace('-', '_')}.csv",
                                mime='text/csv',
                                key="download_predictions_csv_run_pred_main"
                            )
                            st.success(T("run_pred_success_prediction_done"))
                        else:
                            st.error(T("run_pred_error_prediction_failed_internal"))
                    except RuntimeError as e: # Catch errors from make_predictions
                        st.error(str(e)) # Error message already translated by make_predictions
                    except Exception as e:
                        st.error(T("run_pred_error_unexpected", error_message=str(e)))
                        # import traceback
                        # st.error(traceback.format_exc()) # For more detailed debugging
        else:
            st.error(T("run_pred_error_preprocessing_failed", default="Data preprocessing for prediction failed. Check data and logs."))


if __name__ == "__main__":
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    # Mock artifacts for standalone testing (replace with actual loading in app flow)
    if 'gui_artifacts_loaded' not in st.session_state: # Simple flag to load once
        st.session_state.gui_artifacts_loaded = True # Prevent re-mocking on reruns
        # Create dummy artifact files if they don't exist for testing
        models_dir_test = "models"
        os.makedirs(models_dir_test, exist_ok=True)
        if not os.path.exists(os.path.join(models_dir_test, "scaler.joblib")):
            from sklearn.preprocessing import StandardScaler
            dummy_scaler = StandardScaler()
            # Fit on some dummy data that matches expected number of features after CLR
            # This number needs to be known. Let's say it's 35 after CLR on 36 - 1 (if one is dropped)
            # This is tricky for a simple dummy. For now, just save an unfitted one.
            joblib.dump(dummy_scaler, os.path.join(models_dir_test, "scaler.joblib"))

        if not os.path.exists(os.path.join(models_dir_test, "label_encoder_classes.json")):
            with open(os.path.join(models_dir_test, "label_encoder_classes.json"), 'w') as f:
                json.dump(['Au-rich PCDs', 'Cu-rich PCDs'], f) # Example
        if not os.path.exists(os.path.join(models_dir_test, "final_feature_names.json")):
             # These should be names AFTER CLR, e.g. ['SiO2_ppm_clr', 'TiO2_ppm_clr', ...]
            dummy_final_names = [f'feature_{i}_clr' for i in range(len(RAW_EXPECTED_GEOCHEMICAL_FEATURES))]
            with open(os.path.join(models_dir_test, "final_feature_names.json"), 'w') as f:
                json.dump(dummy_final_names, f)

    show_page()