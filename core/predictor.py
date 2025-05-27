# core/predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import torch # For PyTorch
from util.language import T # Import T

# Assuming CLASS_NAMES will be accessible, e.g., from a session_state or config
# For now, predictor itself doesn't need CLASS_NAMES, it returns numeric predictions / probabilities

def make_predictions(model, processed_data_df, model_name, num_classes_for_dnn): # Added num_classes for DNN
    """Makes predictions using the loaded model and processed data."""
    if model is None or processed_data_df is None:
        st.error("Model or processed data is None in make_predictions.") # Should be caught earlier
        return None, None

    try:
        if model_name in ["Random Forest", "XGBoost", "SVM"]:
            # Ensure data is numpy for sklearn models if they expect it (though DataFrames usually work)
            # processed_data_np = processed_data_df.values
            predictions = model.predict(processed_data_df) # Use DataFrame directly
            
            if hasattr(model, "predict_proba"):
                probabilities_all_classes = model.predict_proba(processed_data_df)
                # For binary, probabilities_all_classes is (n_samples, 2). We need prob of positive class (class 1)
                # For multi-class, this will be (n_samples, n_classes)
                # The run_prediction page currently expects single probability for CLASS_NAMES[1]
                # This needs careful handling if we expand to multi-class later.
                # For now, assume binary classification where CLASS_NAMES[1] is the positive class.
                # The label encoder used in training should map CLASS_NAMES[1] to index 1.
                probabilities_positive_class = probabilities_all_classes[:, 1]

            elif hasattr(model, "decision_function"): # For SVM without probability=True
                decision_scores = model.decision_function(processed_data_df)
                # Simple sigmoid scaling for demonstration if true probabilities aren't available
                # This is not a true probability. Consider ensuring SVM is trained with probability=True.
                probabilities_positive_class = 1 / (1 + np.exp(-decision_scores))
                if probabilities_positive_class.ndim == 2 and probabilities_positive_class.shape[1] ==1:
                    probabilities_positive_class = probabilities_positive_class.flatten()
            else:
                # Fallback if no probability mechanism
                probabilities_positive_class = np.full(len(predictions), 0.5) # Placeholder
                st.warning(f"Model {model_name} does not have predict_proba or decision_function. Probabilities are placeholders.")

        elif model_name == "PyTorch DNN":
            # Convert DataFrame to PyTorch tensor
            # Ensure processed_data_df columns are in the exact order the model expects
            # This should be guaranteed by preprocess_data_for_prediction using trained_feature_names
            data_tensor = torch.tensor(processed_data_df.values, dtype=torch.float32).to(model.network[0].weight.device) # Get device from model

            with torch.no_grad():
                outputs = model(data_tensor)

            if num_classes_for_dnn == 2: # Binary classification with BCEWithLogitsLoss
                # Output is raw logits (1 neuron for binary)
                probabilities_positive_class = torch.sigmoid(outputs).cpu().numpy().flatten()
                predictions = (probabilities_positive_class > 0.5).astype(int)
            else: # Multi-class classification with CrossEntropyLoss
                # Output is raw logits (num_classes neurons)
                probabilities_all_classes = torch.softmax(outputs, dim=1).cpu().numpy()
                predictions = np.argmax(probabilities_all_classes, axis=1)
                # For multi-class, how to define 'probabilities_positive_class' depends on what's needed downstream.
                # If a specific class's probability is needed, select it.
                # For now, for consistency with binary, this might need adjustment or downstream handling.
                # Let's assume for now the GUI wants prob of the class mapped to '1' by label encoder.
                if probabilities_all_classes.shape[1] > 1:
                    probabilities_positive_class = probabilities_all_classes[:, 1] # Prob of class index 1
                else: # Should not happen for multi-class > 2
                    probabilities_positive_class = probabilities_all_classes.flatten()


        else:
            raise ValueError(T("predictor_unknown_model_type", model_name=model_name))

        return predictions, probabilities_positive_class

    except Exception as e:
        # Using f-string for model name, T() for base string
        raise RuntimeError(T("run_pred_error_runtime_prediction", model_name=model_name, error_message=str(e))) # Modified key for run_pred context