from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import numpy as np
import joblib
import os



def train_svm(X_train, y_train, feature_names, random_state=42, model_filename="svm_model.joblib",model_save_dir="models"):
    """
    Trains an SVM classifier using GridSearchCV, calculates permutation importances,
    saves the model, and returns the best model and its importances.
    """
    # Create directory for saving models
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, model_filename)

    param_grid = [
        {'C': [0.1, 1, 10], 'kernel': ['linear']}, # Reduced for speed
        {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'gamma': [0.01, 0.1, 'scale']}, # Reduced
        # {'C': [0.1, 1, 10, 100], 'kernel': ['poly'], 'degree': [2, 3], 'gamma': ['scale', 'auto']} # Poly can be slow
    ]
    
    print("\nTuning SVM hyperparameters using GridSearchCV...")
    svm = SVC(probability=True, random_state=random_state) # probability=True for ROC/PR curves
    # Using fewer folds (cv=2) and fewer params for faster execution in this example. Adjust as needed.
    grid_search = GridSearchCV(svm, param_grid, cv=2, scoring='accuracy', verbose=1, n_jobs=-1) 
    
    grid_search.fit(X_train, y_train)
    
    best_svm_model = grid_search.best_estimator_
    print(f"Best SVM parameters: {grid_search.best_params_}")
    print(f"Best SVM cross-validation accuracy: {grid_search.best_score_:.4f}")

    print("Calculating permutation importances for SVM (can be slow)...")
    try:
        # n_repeats=5 for faster execution, default is higher
        perm_importance = permutation_importance(
            best_svm_model, X_train, y_train, n_repeats=5, random_state=random_state, n_jobs=-1 
        )
        importances = perm_importance.importances_mean
    except Exception as e:
        print(f"Could not calculate permutation importance for SVM: {e}. Setting importances to None.")
        importances = np.array([]) # Empty array if fails

    try:
        joblib.dump(best_svm_model, model_save_path)
        print(f"SVM model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving SVM model: {e}")
        
    return best_svm_model, importances