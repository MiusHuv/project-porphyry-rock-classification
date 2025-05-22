# geochem_classifier_gui/core/predictor.py
import pandas as pd
import numpy as np

def make_predictions(model, processed_data, model_name):
    """Makes predictions using the loaded model and processed data."""
    if model is None or processed_data is None:
        return None, None

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
                probabilities = model.decision_function(processed_data)
            else:
                probabilities = np.array([0.0] * len(predictions)) # Placeholder if no proba/decision_func

        elif model_name == "DNN-Keras":
            # Keras model.predict gives probabilities for each class if output is softmax
            # Or a single probability if output is sigmoid for binary classification
            raw_probabilities = model.predict(processed_data)
            if raw_probabilities.shape[1] > 1: # Softmax output
                probabilities = raw_probabilities[:, 1] # Assuming class 1 is 'Cu-rich'
                predictions = np.argmax(raw_probabilities, axis=1)
            else: # Sigmoid output
                probabilities = raw_probabilities.flatten()
                predictions = (probabilities > 0.5).astype(int) # Thresholding at 0.5
            
            # Map predictions (0 or 1) back to labels if needed, or handle this in Streamlit page
            # For now, assuming 0 and 1, will be mapped later.

        else:
            raise ValueError("Unknown model type for prediction.")

        return predictions, probabilities

    except Exception as e:
        # st.error(f"Error during prediction: {e}") # Handled in Streamlit page
        raise RuntimeError(f"Prediction error with {model_name}: {e}")