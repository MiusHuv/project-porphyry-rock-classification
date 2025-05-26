import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import matplotlib.pyplot as plt # For plotting learning curves if desired

MODEL_SAVE_DIR = "trained_models"
PLOT_SAVE_DIR = "output_plots/model_specific" 
if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
if not os.path.exists(PLOT_SAVE_DIR): os.makedirs(PLOT_SAVE_DIR)

def train_xgboost(X_train, y_train, num_classes, feature_names,
                  random_state=42, model_filename="xgboost_model.joblib"):
    """
    Trains an XGBoost classifier using GridSearchCV, saves the model,
    and returns the best model and its feature importances.
    """
    print("\nTraining XGBoost model...")

    # Parameter grid for GridSearchCV
    # Keep it small for faster execution, expand for better tuning
    param_grid = {
        'n_estimators': [100, 200], # Number of boosting rounds
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8]
    }

    # XGBoost classifier setup
    # For multi-class, objective is 'multi:softprob', eval_metric often 'mlogloss' or 'merror'
    # For binary, objective is 'binary:logistic', eval_metric 'logloss' or 'error'
    if num_classes == 2:
        objective = 'binary:logistic'
        eval_metric = 'logloss' # or 'auc'
        # For binary classification, y_train should be 0 or 1.
        # LabelEncoder usually handles this if original labels are not 0/1.
    else:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss' 
    
    xgb_model = xgb.XGBClassifier(
        objective=objective,
        eval_metric=eval_metric,
        random_state=random_state,
        n_jobs=-1 # Use all available cores
    )

    # Split training data for early stopping in GridSearchCV if not supported directly
    # Or rely on GridSearchCV's own CV for evaluation
    # For XGBoost, early stopping is typically handled during model.fit() if an eval_set is provided.
    # GridSearchCV does not directly use that fit's early stopping for its meta-estimation.
    # We can use a simpler GridSearchCV here.

    print("Tuning XGBoost hyperparameters using GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy', # or 'roc_auc' for binary, 'neg_log_loss'
        cv=2, # 2-fold CV for speed, increase for robustness (e.g., 3 or 5)
        verbose=1,
        n_jobs=-1 # Use all available cores for grid search
    )

    grid_search.fit(X_train, y_train) # y_train should be numerically encoded

    best_xgb_model = grid_search.best_estimator_
    print(f"Best XGBoost parameters: {grid_search.best_params_}")
    print(f"Best XGBoost cross-validation accuracy: {grid_search.best_score_:.4f}")

    # Feature importances
    importances = best_xgb_model.feature_importances_

    # Save the model
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    try:
        joblib.dump(best_xgb_model, model_save_path)
        print(f"XGBoost model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving XGBoost model: {e}")
        
    # Optional: Plotting learning curve or tree (can be complex)
    # xgb.plot_importance(best_xgb_model, max_num_features=10)
    # plt.savefig(os.path.join(PLOT_SAVE_DIR, "xgb_feature_importances_native.png"))
    # plt.close()

    return best_xgb_model, importances