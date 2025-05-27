# core/model_loader.py
import joblib
import xgboost as xgb
import torch # For PyTorch
import json # For loading JSON configs
import os
import streamlit as st
from util.language import T
from pathlib import Path
# Import the DNN model class definition
from core.train_dnn_model import SimpleDNN, DEVICE # Ensure DEVICE is also imported or defined

MODEL_DIR = "models" # Adjust path as needed


# Global cache for preprocessor parts
@st.cache_resource
def load_gui_preprocessor_artifacts():
    artifacts = {
        "scaler": None,
        "label_encoder_classes": None,
        "final_feature_names": None
    }
    model_base_dir = Path(__file__).parent.parent / "models"
    scaler_path = model_base_dir / "scaler.joblib"
    le_classes_path = model_base_dir / "label_encoder_classes.json"
    feature_names_path = model_base_dir / "final_feature_names.json"

    try:
        if scaler_path.exists():
            artifacts["scaler"] = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}") # Debug log
        else:
            st.error(T("model_loader_error_loading_artifact", artifact_name="Scaler", file_path=scaler_path)) # New key
            print(f"Scaler file not found at {scaler_path}") # Debug log
    except Exception as e:
        st.error(T("model_loader_error_generic_artifact", artifact_name="Scaler", error_message=str(e))) # New key

    try:
        if le_classes_path.exists():
            with open(le_classes_path, 'r') as f:
                artifacts["label_encoder_classes"] = json.load(f)
            print(f"Label encoder classes loaded from {le_classes_path}") # Debug log
        else:
            st.error(T("model_loader_error_loading_artifact", artifact_name="Label Encoder Classes", file_path=le_classes_path))
            print(f"Label encoder classes file not found at {le_classes_path}") # Debug log
    except Exception as e:
        st.error(T("model_loader_error_generic_artifact", artifact_name="Label Encoder Classes", error_message=str(e)))

    try:
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                artifacts["final_feature_names"] = json.load(f)
            print(f"Final feature names loaded from {feature_names_path}") # Debug log
        else:
            st.error(T("model_loader_error_loading_artifact", artifact_name="Final Feature Names", file_path=feature_names_path))
            print(f"Final feature names file not found at {feature_names_path}") # Debug log
    except Exception as e:
        st.error(T("model_loader_error_generic_artifact", artifact_name="Final Feature Names", error_message=str(e)))

    if not all(artifacts.values()): # If any failed to load
        return None # Indicate failure
    return artifacts


def load_selected_model(model_name):
    model_base_dir = Path(__file__).parent.parent / "models"
    model_path_map = {
        "Random Forest": model_base_dir / "random_forest_model.joblib",
        "XGBoost": model_base_dir / "xgboost_model.joblib",
        "SVM": model_base_dir / "svm_model.joblib",
        "PyTorch DNN": {
            "state_dict": model_base_dir / "pytorch_dnn_model.pth",
            "config": model_base_dir / "pytorch_dnn_model_config.json"
        }
    }
    model_info_or_path = model_path_map.get(model_name)

    if not model_info_or_path:
        st.error(T("model_loader_invalid_model_name", model_name=model_name))
        return None

    try:
        if model_name in ["Random Forest", "SVM", "XGBoost"]: # XGBoost now loaded with joblib
            model_path = model_info_or_path # It's a Path object
            if not model_path.exists():
                st.error(T("model_loader_file_not_found", model_name=model_name, file_path=str(model_path)))
                return None
            model = joblib.load(model_path) # Use joblib.load for these models

        elif model_name == "PyTorch DNN":
            pth_path = model_info_or_path["state_dict"]
            config_path = model_info_or_path["config"]
            if not pth_path.exists() or not config_path.exists():
                st.error(T("model_loader_dnn_files_missing", pth_path=str(pth_path), config_path=str(config_path)))
                return None
            with open(config_path, 'r') as f:
                config = json.load(f)
            model = SimpleDNN(
                input_dim=config['input_dim'],
                num_classes=config['num_classes'] if config['num_classes'] > 2 else 1, 
                hidden_layers_config=config['hidden_layers_config'],
                dropout_rates=config['dropout_rates']
            ).to(DEVICE)
            model.load_state_dict(torch.load(pth_path, map_location=DEVICE))
            model.eval()
        else:
            st.error(T("model_loader_invalid_model_name", model_name=model_name))
            return None
        # st.success(T("model_loader_success_model_load", model_name=model_name)) # Optional
        return model
    except Exception as e:
        st.error(T("model_loader_error_loading_model", model_name=model_name, error_message=str(e)))
        if model_name == "XGBoost":
            import traceback
            st.error(f"XGBoost loading traceback: {traceback.format_exc()}")
        return None

# `load_preprocessor_object` is replaced by `load_gui_preprocessor_artifacts` which returns a dict.
# Pages will call `load_gui_preprocessor_artifacts` and access `artifacts["scaler"]`, etc.