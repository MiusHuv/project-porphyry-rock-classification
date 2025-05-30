from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Import from your project structure
from core.visualizer import plot_loss_vs_trees

def train_random_forest(X_train, y_train, feature_names, 
                        n_estimators_range=None, random_state=42, 
                        plot_curves=True, model_filename="random_forest_model.joblib",
                        model_save_dir="models", # New parameter with default
                        plot_save_dir="assets/eda_plots/model_specific"): # New parameter
    """
    Trains a Random Forest classifier, finds optimal number of trees,
    saves the model, and returns the best model and its feature importances.
    """
    if n_estimators_range is None:
        n_estimators_range = list(range(50, 201, 25)) # Default range

    # Create directory for saving models if it doesn't exist
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, model_filename)

    # Split training data further to get a validation set for n_estimators tuning
    if len(X_train) > 100 and len(np.unique(y_train)) > 1 : # Ensure enough samples and classes for stratification
         X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
             X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
         )
    else: 
        print("Warning: Training data size is small or only one class for stratification. Using full training set for n_estimators curve (less ideal).")
        X_train_part, y_train_part = X_train, y_train
        # For validation, use OOB score if no separate validation set, or fall back to training score
        X_val_part, y_val_part = X_train, y_train 

    train_scores = []
    val_scores = []

    print(f"\nTuning n_estimators for Random Forest (range: {min(n_estimators_range)} to {max(n_estimators_range)})...")
    for n_trees in n_estimators_range:
        model = RandomForestClassifier(n_estimators=n_trees, random_state=random_state, 
                                       oob_score=(X_val_part is X_train_part and len(X_train_part) > 0) ) # Enable OOB if no val split and data exists
        model.fit(X_train_part, y_train_part)
        
        train_pred = model.predict(X_train_part)
        train_scores.append(accuracy_score(y_train_part, train_pred))
        
        if X_val_part is not X_train_part: # Proper validation set
            val_pred = model.predict(X_val_part)
            val_scores.append(accuracy_score(y_val_part, val_pred))
        elif hasattr(model, 'oob_score_') and model.oob_score_ is not None: # Using OOB score
             val_scores.append(model.oob_score_)
        else: # Fallback to training score if OOB not available/applicable
             val_scores.append(accuracy_score(y_train_part, train_pred))


    if not val_scores: # Should not happen if n_estimators_range is not empty
        print("Error: Could not compute validation scores for Random Forest. Skipping optimal tree selection.")
        optimal_n_trees = n_estimators_range[0] # Default to first value
    else:
        optimal_n_trees_idx = np.argmax(val_scores)
        optimal_n_trees = n_estimators_range[optimal_n_trees_idx]

    if plot_curves:
        loss_fig, optimal_n_trees_from_plot = plot_loss_vs_trees(train_scores, val_scores, n_estimators_range, "Random Forest")
    else:
        print(f"Optimal number of trees for Random Forest based on validation accuracy: {optimal_n_trees}")
        loss_fig, optimal_n_trees_from_plot = None, n_estimators_range[np.argmax(val_scores)] if val_scores else n_estimators_range[0]

    print(f"Training final Random Forest model with {optimal_n_trees} trees...")
    best_rf_model = RandomForestClassifier(n_estimators=optimal_n_trees, random_state=random_state, oob_score=True)
    best_rf_model.fit(X_train, y_train) # Train on the full training set
    
    importances = best_rf_model.feature_importances_

    if plot_curves and loss_fig: # loss_fig from plot_loss_vs_trees
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        try:
            loss_fig_path = os.path.join(plot_save_dir, f"rf_loss_vs_trees_{model_filename.split('.')[0]}.png") # Make filename unique
            loss_fig.savefig(loss_fig_path)
            print(f"RF loss vs trees plot saved to {loss_fig_path}")
            plt.close(loss_fig)
        except Exception as e:
            print(f"Error saving RF loss_vs_trees plot: {e}")
            if loss_fig: plt.close(loss_fig)
    
    optimal_n_trees = optimal_n_trees_from_plot # Use the one from the plot function

    try:
        joblib.dump(best_rf_model, model_save_path)
        print(f"Random Forest model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving Random Forest model: {e}")
    
    return best_rf_model, importances