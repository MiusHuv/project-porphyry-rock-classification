# geochem_classifier_gui/core/model_loader.py
import joblib
import xgboost as xgb
# from tensorflow import keras # Or from keras.models import load_model
import streamlit as st
import os

MODEL_DIR = "models" # Define your model directory

# Ensure the model directory exists or handle errors
if not os.path.exists(MODEL_DIR):
    # This is a critical error for the app to function.
    # In a real app, you might have a setup script or better error handling.
    print(f"CRITICAL ERROR: Model directory '{MODEL_DIR}' not found!")
    # st.error(f"Model directory '{MODEL_DIR}' not found. Please ensure models are in place.")


def load_selected_model(model_name):
    """Loads a specific pre-trained model."""
    model_path_map = {
        "Random Forest": os.path.join(MODEL_DIR, "random_forest_model.joblib"),
        "XGBoost": os.path.join(MODEL_DIR, "xgboost_model.json"), # or .ubj, .pkl etc.
        "SVM": os.path.join(MODEL_DIR, "svm_model.joblib"),
        "DNN-Keras": os.path.join(MODEL_DIR, "dnn_keras_model.h5")
    }
    model_file = model_path_map.get(model_name)

    if not model_file or not os.path.exists(model_file):
        st.error(f"Model file for {model_name} not found at {model_file}. Please ensure models are correctly placed.")
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
        # elif model_name == "DNN-Keras":
        #     model = keras.models.load_model(model_file)
        else:
            st.error("Invalid model name selected.")
            return None
        # st.success(f"{model_name} model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading {model_name} model: {e}")
        return None

def load_preprocessor_object():
    """Loads the saved preprocessor object (e.g., scaler, CLR transformer)."""
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    if not os.path.exists(preprocessor_path):
        st.error(f"Preprocessor file not found at {preprocessor_path}. This is critical for predictions.")
        return None
    try:
        preprocessor = joblib.load(preprocessor_path)
        # st.info("Preprocessor loaded successfully.")
        return preprocessor
    except Exception as e:
        st.error(f"Error loading preprocessor: {e}")
        return None