# geochem_classifier_gui/core/model_loader.py
import joblib
import xgboost as xgb
# from tensorflow import keras # Or from keras.models import load_model
import streamlit as st
import os
from util.language import T # Import T

MODEL_DIR = "models" # Define your model directory

# Ensure the model directory exists or handle errors
if not os.path.exists(MODEL_DIR):
    # This is a critical error for the app to function.
    # This print is for the console, not the Streamlit UI.
    # If you want a UI error, you'd need to raise an exception or handle it in the app's main flow.
    print(f"CRITICAL ERROR: Model directory '{MODEL_DIR}' not found!")
    # For a UI error, this would need to be handled where functions from this module are called,
    # as st.error() might not be appropriate at module load time if it causes issues.
    # A possible approach is to have functions return a specific error status or raise an exception.


def load_selected_model(model_name):
    """Loads a specific pre-trained model."""
    model_path_map = {
        "Random Forest": os.path.join(MODEL_DIR, "random_forest_model.joblib"),
        "XGBoost": os.path.join(MODEL_DIR, "xgboost_model.json"), # or .ubj, .pkl etc.
        "SVM": os.path.join(MODEL_DIR, "svm_model.joblib"),
        "DNN-Keras": os.path.join(MODEL_DIR, "dnn_keras_model.h5") # Assuming you will add Keras loading logic
    }
    model_file = model_path_map.get(model_name)

    if not model_file or not os.path.exists(model_file):
        st.error(T("model_loader_file_not_found", model_name=model_name, file_path=str(model_file), default=f"Model file for {model_name} not found at {model_file}. Please ensure models are correctly placed."))
        return None

    try:
        if model_name == "Random Forest" or model_name == "SVM":
            model = joblib.load(model_file)
        elif model_name == "XGBoost":
            # For scikit-learn wrapper:
            # model = joblib.load(model_file) 
            # For native XGBoost:
            model = xgb.Booster() # Or xgb.XGBClassifier() if you saved the scikit-learn wrapper
            model.load_model(model_file) # If saved XGBoost model
            # If you saved an XGBClassifier object with joblib:
            # model = joblib.load(model_file)
        elif model_name == "DNN-Keras":
            # from tensorflow import keras # Ensure Keras is imported if you use this
            # model = keras.models.load_model(model_file)
            # For now, as Keras is commented out, let's treat it as an unloaded model type
            st.error(T("model_loader_dnn_not_implemented", default="DNN-Keras model loading is not yet fully implemented in model_loader.py."))
            return None
        else:
            # This case should ideally not be reached if model_name is from a controlled list
            st.error(T("model_loader_invalid_model_name", model_name=model_name, default=f"Invalid model name selected: {model_name}."))
            return None
        # st.success(T("model_loader_success_model_load", model_name=model_name, default=f"{model_name} model loaded successfully.")) # If you want success messages
        return model
    except Exception as e:
        st.error(T("model_loader_error_loading_model", model_name=model_name, error_message=str(e), default=f"Error loading {model_name} model: {e}"))
        return None

def load_preprocessor_object():
    """Loads the saved preprocessor object (e.g., scaler, CLR transformer)."""
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    if not os.path.exists(preprocessor_path):
        st.error(T("model_loader_preprocessor_not_found", file_path=preprocessor_path, default=f"Preprocessor file not found at {preprocessor_path}. This is critical for predictions."))
        return None
    try:
        preprocessor = joblib.load(preprocessor_path)
        # st.info(T("model_loader_success_preprocessor_load", default="Preprocessor loaded successfully.")) # If you want info messages
        return preprocessor
    except Exception as e:
        st.error(T("model_loader_error_loading_preprocessor", error_message=str(e), default=f"Error loading preprocessor: {e}"))
        return None