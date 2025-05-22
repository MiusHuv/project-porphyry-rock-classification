# geochem_classifier_gui/core/data_handler.py
import pandas as pd
import streamlit as st

# These would be the columns your models were trained on
# Ensure the order is consistent if your model expects a specific order
EXPECTED_COLUMNS = ['SiO2', 'TiO2', 'Al2O3', 'TFe2O3', 'MnO', 'MgO', 'CaO', 
                    'Na2O', 'K2O', 'P2O5', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 
                    'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 
                    'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Th', 'U'] # 36 features

def load_data(uploaded_file):
    """Loads data from uploaded CSV or XLSX file."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def validate_data(df):
    """Validates the dataframe for required columns and numeric types."""
    if df is None:
        return None, "No data loaded."

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        return None, f"Missing required columns: {', '.join(missing_cols)}. Please ensure your file contains all 36 features."

    # Select only the expected feature columns for further processing/prediction
    feature_df = df[EXPECTED_COLUMNS].copy()

    # Check for non-numeric data - simplistic check, might need more robust handling
    for col in EXPECTED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(feature_df[col]):
            try:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='raise')
            except ValueError:
                return None, f"Column '{col}' contains non-numeric data that could not be converted. Please clean your data."
    
    # As per project: "handle missing values (zero replacement for CLR)"
    # This zero replacement should happen *before* CLR if CLR is part of your preprocessor.
    # If your preprocessor (e.g., a scikit-learn pipeline) handles imputation, this might be done there.
    # For simplicity, let's assume zero replacement for features before any transformation.
    # THIS IS A CRITICAL STEP and must match your training pipeline.
    feature_df.fillna(0, inplace=True) # Example: Zero replacement

    return feature_df, "Data validation successful."


def preprocess_data_for_prediction(df, preprocessor):
    """
    Applies the *exact* same preprocessing steps as used during training.
    The 'preprocessor' object should be loaded from models/preprocessor.joblib
    This might include CLR transformation, scaling, etc.
    """
    if df is None or preprocessor is None:
        return None
    try:
        # Example: if preprocessor is a scikit-learn transformer or pipeline
        processed_df = preprocessor.transform(df)
        # Ensure the output is a DataFrame with correct feature names if subsequent steps rely on it
        # Or if the model expects a NumPy array, ensure that format.
        # This depends heavily on how your preprocessor and models are set up.
        if not isinstance(processed_df, pd.DataFrame) and hasattr(preprocessor, 'get_feature_names_out'):
             processed_df = pd.DataFrame(processed_df, columns=preprocessor.get_feature_names_out())
        elif not isinstance(processed_df, pd.DataFrame):
             processed_df = pd.DataFrame(processed_df, columns=df.columns) # Fallback if no feature names from preprocessor

        return processed_df
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        return None