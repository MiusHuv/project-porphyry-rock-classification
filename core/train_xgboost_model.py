import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import matplotlib.pyplot as plt # For plotting learning curves if desired
from pathlib import Path


def train_xgboost(X_train, y_train, num_classes, feature_names,
                  random_state=42, model_filename="xgboost_model.joblib",
                  model_save_dir="models", plot_curves=True,# New parameter with default
                  plot_save_dir="assets/eda_plots/model_specific"): # New parameter
    """
    Trains an XGBoost classifier using GridSearchCV, saves the model,
    and returns the best model and its feature importances.
    """
    print("\nTraining XGBoost model...")

    # Parameter grid for GridSearchCV
    # Keep it small for faster execution, expand for better tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }
    if num_classes == 2:
        objective = 'binary:logistic'
        eval_metric = 'logloss'
    else:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss'

    xgb_model_instance = xgb.XGBClassifier(
        objective=objective,
        eval_metric=eval_metric,
        random_state=random_state,
        n_jobs=-1
    )
    print("Tuning XGBoost hyperparameters using GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=xgb_model_instance,
        param_grid=param_grid,
        scoring='accuracy',
        cv=2,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_xgb_model = grid_search.best_estimator_
    print(f"Best XGBoost parameters: {grid_search.best_params_}")
    print(f"Best XGBoost cross-validation accuracy: {grid_search.best_score_:.4f}")

    importances = best_xgb_model.feature_importances_

    # Save the model using joblib
    model_save_path_obj = Path(model_save_dir) / model_filename # Use Path
    model_save_path_obj.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    try:
        joblib.dump(best_xgb_model, model_save_path_obj) # Use joblib.dump
        print(f"XGBoost model saved to {model_save_path_obj}")
    except Exception as e:
        print(f"Error saving XGBoost model with joblib: {e}")
        
    return best_xgb_model, importances