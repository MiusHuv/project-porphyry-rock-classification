# geochem_classifier_gui/core/predictor.py
import pandas as pd
import numpy as np

from util.language import T # Import T

def make_predictions(model, processed_data, model_name):
    """Makes predictions using the loaded model and processed data."""
    if model is None or processed_data is None:
        return None, None # Error should be handled by caller

    try:
        if model_name in ["Random Forest", "XGBoost", "SVM"]:
            # Scikit-learn compatible models often have predict and predict_proba
            predictions = model.predict(processed_data)
            # For probabilities, usually [prob_class_0, prob_class_1]
            # We need the probability of the positive class (e.g., 'Cu-rich')
            # Assuming 'Cu-rich' is class 1 after label encoding during training
            # This needs to be consistent with your training!
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(processed_data)[:, 1] # Prob of positive class
            elif hasattr(model, "decision_function"): # For SVM without probability=True
                # Decision function scores might not be probabilities but can be used as confidence
                # To make it behave somewhat like probability (0-1 range), you might need to scale it,
                # but for now, returning raw scores. Or, ensure SVM has probability=True.
                decision_scores = model.decision_function(processed_data)
                # Simple sigmoid scaling for demonstration, not a true probability
                # probabilities = 1 / (1 + np.exp(-decision_scores)) 
                # For simplicity, if not true probabilities, it might be better to indicate this or handle upstream
                # For now, let's assume if predict_proba is not there, we might not have reliable probabilities.
                # Or, the calling function should be aware of what decision_function returns.
                # Let's return raw decision scores if predict_proba is absent for SVM.
                # The calling code (run_prediction.py) will need to handle this if it expects probabilities.
                # For now, returning a placeholder or the decision scores directly.
                # For consistency, let's return a placeholder if true probabilities are not available.
                probabilities = np.array([0.5] * len(predictions)) # Placeholder
                # A better placeholder might be just the decision scores if the model is SVM
                if model_name == "SVM":
                     probabilities = model.decision_function(processed_data) # Return decision scores for SVM
                else: # For other models if predict_proba is missing (unlikely for RF/XGB)
                     probabilities = np.array([np.nan] * len(predictions))


            else:
                probabilities = np.array([np.nan] * len(predictions)) # Placeholder if no proba/decision_func

        elif model_name == "DNN-Keras":
            # Ensure Keras is imported and model is a Keras model
            # from tensorflow import keras
            # if not isinstance(model, keras.Model):
            #     raise ValueError(T("predictor_dnn_invalid_model_type", default="DNN-Keras model is not a valid Keras model instance."))
            
            raw_probabilities = model.predict(processed_data)
            if raw_probabilities.ndim > 1 and raw_probabilities.shape[1] > 1: # Softmax output for multi-class (even if used for binary)
                probabilities = raw_probabilities[:, 1] # Assuming class 1 is 'Cu-rich'
                predictions = np.argmax(raw_probabilities, axis=1)
            else: # Sigmoid output for binary classification
                probabilities = raw_probabilities.flatten()
                predictions = (probabilities > 0.5).astype(int) # Thresholding at 0.5
            
        else:
            # This error will be raised if model_name is not recognized
            raise ValueError(T("predictor_unknown_model_type", model_name=model_name, default=f"Unknown model type for prediction: {model_name}."))

        return predictions, probabilities

    except Exception as e:
        # Errors during the actual prediction step are often critical.
        # It's better to let them propagate or be caught by the calling Streamlit page
        # so a user-friendly message can be displayed there.
        # For example, run_prediction.py can catch this and show st.error().
        # Adding a generic error here might be redundant if the caller handles it.
        # If we do handle it here, it should be a clear message.
        # For now, re-raising to be handled by the caller, which is good practice.
        # Or return None, None and log the error
        raise RuntimeError(f"Prediction error with {model_name}: {e}")