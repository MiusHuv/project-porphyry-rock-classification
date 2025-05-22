# geochem_classifier_gui/pages/2_🚀_Run_Prediction.py
import streamlit as st
import pandas as pd
from core.data_handler import load_data, validate_data, preprocess_data_for_prediction, EXPECTED_COLUMNS
from core.model_loader import load_selected_model, load_preprocessor_object
from core.predictor import make_predictions
from core.visualizer import CLASS_NAMES # To map numeric predictions back to labels

def show_page():
    st.title("🚀 Run Prediction on New Samples")

    # --- File Upload and Model Selection ---
    st.sidebar.header("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload your sample data (.csv or .xlsx)", type=['csv', 'xlsx'])
    
    model_options = ["Random Forest", "XGBoost", "SVM", "DNN-Keras"]
    selected_model_name = st.sidebar.selectbox("Choose a classification model:", model_options)

    # --- Load Preprocessor ---
    # Cache the preprocessor loading
    @st.cache_resource 
    def get_preprocessor():
        return load_preprocessor_object()
    
    preprocessor = get_preprocessor()
    if preprocessor is None:
        st.error("Critical error: Preprocessor could not be loaded. Predictions cannot proceed.")
        st.stop()

    # --- Load Model ---
    # Cache model loading based on name
    @st.cache_resource
    def get_model(model_name):
        return load_selected_model(model_name)

    model = get_model(selected_model_name)
    if model is None:
        st.sidebar.error(f"Model {selected_model_name} could not be loaded.")
        st.stop()
    else:
        st.sidebar.success(f"{selected_model_name} model ready.")

    # --- Data Handling & Prediction ---
    if uploaded_file is not None:
        raw_df = load_data(uploaded_file)

        if raw_df is not None:
            st.subheader("1. Uploaded Data Preview (First 5 Rows)")
            st.dataframe(raw_df.head())

            # Allow user to specify true label column if present for later evaluation
            true_label_col_name = None
            if st.checkbox("My data includes a 'True Label' column for performance evaluation"):
                # Try to guess or let user pick
                potential_label_cols = [col for col in raw_df.columns if col not in EXPECTED_COLUMNS]
                if not potential_label_cols:
                    potential_label_cols = list(raw_df.columns) # Fallback if only feature columns are present
                
                if potential_label_cols:
                    true_label_col_name = st.selectbox(
                        "Select your 'True Label' column:", 
                        options=potential_label_cols, 
                        index=0,
                        help="This column will be used for performance metrics if available."
                    )
                    # Store for potential use in Performance Visualizer page
                    st.session_state.true_label_column = true_label_col_name 
                    st.session_state.raw_df_with_labels = raw_df # Store the full df
                else:
                    st.warning("Could not identify a potential label column.")


            st.subheader("2. Data Validation & Preprocessing")
            feature_df, validation_msg = validate_data(raw_df.copy()) # Use a copy
            
            if feature_df is None:
                st.error(f"Validation failed: {validation_msg}")
            else:
                st.success(validation_msg)
                
                processed_df_for_prediction = preprocess_data_for_prediction(feature_df, preprocessor)

                if processed_df_for_prediction is not None:
                    st.write("Data preprocessed successfully for prediction.")
                    # st.dataframe(processed_df_for_prediction.head()) # Optional: show processed data

                    st.subheader("3. Run Prediction")
                    if st.button(f"Predict using {selected_model_name}", type="primary"):
                        with st.spinner("Running predictions..."):
                            try:
                                predictions_numeric, probabilities = make_predictions(model, processed_df_for_prediction, selected_model_name)
                                
                                if predictions_numeric is not None and probabilities is not None:
                                    # Combine results with original data (or selected columns from it)
                                    results_df = raw_df.copy() # Start with original data
                                    
                                    # Map numeric predictions to class labels
                                    # Ensure CLASS_NAMES[0] is Au-rich and CLASS_NAMES[1] is Cu-rich as per project.
                                    # This mapping MUST align with your training label encoding.
                                    results_df['Predicted Class'] = [CLASS_NAMES[p] for p in predictions_numeric]
                                    results_df['Prediction Probability (Cu-rich)'] = probabilities 
                                    # (Assuming probability is for the 'Cu-rich' class)

                                    st.session_state.predictions_df = results_df # Save for download and performance page

                                    st.subheader("4. Prediction Results")
                                    st.dataframe(results_df)

                                    # Summary
                                    cu_rich_count = (results_df['Predicted Class'] == CLASS_NAMES[1]).sum()
                                    au_rich_count = (results_df['Predicted Class'] == CLASS_NAMES[0]).sum()
                                    st.metric(label="Predicted Cu-rich Samples", value=cu_rich_count)
                                    st.metric(label="Predicted Au-rich Samples", value=au_rich_count)

                                    # Download button
                                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Download Predictions as CSV",
                                        data=csv_data,
                                        file_name=f"predictions_{selected_model_name.replace(' ', '_').lower()}.csv",
                                        mime='text/csv',
                                    )
                                else:
                                    st.error("Prediction failed. Check logs or model compatibility.")
                            except RuntimeError as e:
                                st.error(f"An error occurred during prediction: {e}")
                            except Exception as e:
                                st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Awaiting data file upload...")

if __name__ == "__main__":
    # Initialize session state variables if not already present
    if 'predictions_df' not in st.session_state:
        st.session_state.predictions_df = None
    if 'true_label_column' not in st.session_state:
        st.session_state.true_label_column = None
    if 'raw_df_with_labels' not in st.session_state:
        st.session_state.raw_df_with_labels = None
