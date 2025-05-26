import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import os
try:
    import shap
    SHAP_AVAILABLE = True
    # For PyTorch, shap might need specific handling or newer versions
    # For basic tree models, it's usually straightforward.
    # shap.initjs() # Not needed for saving plots, only for notebook display
except ImportError:
    print("SHAP library not found. Please install with: pip install shap. SHAP plots will be skipped.")
    SHAP_AVAILABLE = False

def plot_loss_vs_trees(train_scores, val_scores, n_estimators_range, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_estimators_range, train_scores, label="Training Score", marker='o', linestyle='-') # T("Training Score")
    ax.plot(n_estimators_range, val_scores, label="Validation Score", marker='o', linestyle='-') # T("Validation Score")
    ax.set_xlabel("Number of Trees") # T("Number of Trees")
    ax.set_ylabel("Accuracy Score") # T("Accuracy Score")
    ax.set_title(f"Score vs. Number of Trees for {model_name}") # T("Score vs. Number of Trees for {model_name}")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    optimal_n_trees_idx = np.argmax(val_scores)
    optimal_n_trees = n_estimators_range[optimal_n_trees_idx]
    print(f"Optimal number of trees for {model_name} based on validation score: {optimal_n_trees} (Validation Score: {val_scores[optimal_n_trees_idx]:.4f})")
    # T("Justification: This is the number of trees where the validation score is maximized...")
    print("Justification: This is the number of trees where the validation score is maximized, "
          "aiming for a balance between model complexity and generalization performance before potential overfitting.")
    return fig, optimal_n_trees # Return fig and optimal_n_trees

def plot_feature_importances(importances, feature_names, model_name, top_n=20):
    if importances is None or len(importances) == 0:
        print(f"No feature importances available or feature_names mismatch for {model_name}.")
        return None
    if len(importances) != len(feature_names):
        print(f"Warning for {model_name}: Mismatch between number of importances ({len(importances)}) and feature_names ({len(feature_names)}). Plotting skipped.")
        return None

    importances_series = pd.Series(importances, index=feature_names)
    sorted_importances = importances_series.sort_values(ascending=False)
    top_n = min(top_n, len(sorted_importances))
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 1.8))) # Adjusted height
    sns.barplot(x=sorted_importances.head(top_n).values, 
                y=sorted_importances.head(top_n).index, 
                hue=sorted_importances.head(top_n).index,
                legend=False,
                ax=ax)
    ax.set_xlabel("Importance") # T("Importance")
    ax.set_ylabel("Feature") # T("Feature")
    ax.set_title(f"Top {top_n} Feature Importances for {model_name}") # T("Top {top_n} Feature Importances for {model_name}")
    plt.tight_layout()
    return fig

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names_list, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names_list, yticklabels=class_names_list, ax=ax)
    ax.set_xlabel("Predicted Label") # T("Predicted Label")
    ax.set_ylabel("True Label") # T("True Label")
    ax.set_title(f"Confusion Matrix for {model_name}") # T("Confusion Matrix for {model_name}")
    plt.tight_layout()
    return fig

def generate_and_save_shap_plots(model, X_sample_df_orig, feature_names_list, model_type_str, num_classes_model, class_names_list=None, plot_dir="output_plots/shap_plots"):
    if not SHAP_AVAILABLE:
        print("SHAP library not available. Skipping SHAP plot generation.")
        return

    # Work on a copy to avoid modifying the original DataFrame passed from pipeline
    X_sample_df = X_sample_df_orig.copy()

    if X_sample_df.empty:
        print(f"SHAP ({model_type_str}): X_sample_df is empty. Skipping SHAP plots.")
        return

    # Ensure X_sample_df has feature_names_list as columns and in the correct order
    if not isinstance(X_sample_df, pd.DataFrame):
        X_sample_df = pd.DataFrame(X_sample_df, columns=feature_names_list)
    else:
        try:
            X_sample_df = X_sample_df[feature_names_list] # Reorder/subset to match feature_names_list
        except KeyError as e:
            print(f"SHAP ({model_type_str}) Error: Columns in X_sample_df do not perfectly match feature_names_list. {e}")
            print(f"X_sample_df columns: {X_sample_df.columns.tolist()}")
            print(f"Expected feature_names_list: {feature_names_list}")
            return
            
    # Ensure column names are strings, especially for TreeExplainer
    X_sample_df.columns = [str(col) for col in feature_names_list]
    # Also ensure the feature_names_list itself contains strings
    feature_names_list_str = [str(fn) for fn in feature_names_list]


    print(f"\nGenerating SHAP plots for {model_type_str}...")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    try:
        explainer = None
        shap_values = None

        if model_type_str.lower() in ['random forest', 'xgboost']:
            explainer = shap.TreeExplainer(model)
            # TreeExplainer might benefit from seeing the data it's explaining for background/expected value,
            # especially if the model has internal structures sensitive to feature distributions.
            # shap_values = explainer.shap_values(X_sample_df, check_additivity=False) # <-- Original
            # Let's try with explainer(X_sample_df) which is the newer API style for TreeExplainer
            # that returns a shap.Explanation object, then .values
            explanation_object = explainer(X_sample_df)
            shap_values = explanation_object.values # This should be (n_samples, n_features, n_classes) for multi-class
                                                    # or (n_samples, n_features) for binary

            # If shap_values from explainer(X) is 3D for multi-class (n_samples, n_features, n_classes),
            # we need to transpose it or handle it for summary/dependence plots which often expect
            # a list of (n_samples, n_features) arrays for multi-class.
            # The explainer.shap_values(X) method usually returns the list directly.
            # Let's stick to explainer.shap_values(X) for more predictable output structure.
            shap_values = explainer.shap_values(X_sample_df, check_additivity=False)


        elif model_type_str.lower() == 'svm' and hasattr(model, 'predict_proba'):
            # ... (SVM KernelExplainer logic as before) ...
            print(f"SHAP ({model_type_str}): Preparing KernelExplainer...")
            X_kernel_sample = X_sample_df
            if len(X_sample_df) > 50:
                X_kernel_sample = shap.sample(X_sample_df, 50, random_state=42)
            
            def svm_predict_proba_for_shap(data_for_svm_np): # KernelExplainer usually passes numpy
                # Model was trained on DataFrame with feature names.
                # We need to convert numpy back to DataFrame with correct feature names for model.predict_proba
                data_for_svm_df = pd.DataFrame(data_for_svm_np, columns=feature_names_list_str)
                probas = model.predict_proba(data_for_svm_df)
                return probas

            explainer = shap.KernelExplainer(svm_predict_proba_for_shap, X_kernel_sample)
            print(f"SHAP ({model_type_str}): Calculating SHAP values with KernelExplainer (this may take a while)...")
            shap_values = explainer.shap_values(X_sample_df, nsamples='auto', check_additivity=False) # 'auto' nsamples for Kernel
            print(f"SHAP ({model_type_str}): SHAP values calculated.")


        else:
            print(f"SHAP ({model_type_str}): Plot generation not specifically configured. Skipping.")
            return

        # --- Summary Plot (Beeswarm) ---
        # Ensure X_sample_df passed to summary_plot has string columns if shap_values were derived using string names
        X_sample_df_for_plot = X_sample_df.copy()
        X_sample_df_for_plot.columns = feature_names_list_str

        if isinstance(shap_values, list): # Multi-class output (list of arrays, one per class)
            for i in range(len(shap_values)):
                plt.figure() # Create a new figure for each plot
                class_label = class_names_list[i] if class_names_list and i < len(class_names_list) else f"Class_{i}"
                shap.summary_plot(shap_values[i], X_sample_df_for_plot, feature_names=feature_names_list_str, show=False, plot_type="dot")
                plt.title(f"SHAP Summary for {class_label} - {model_type_str}")
                plt.savefig(os.path.join(plot_dir, f"{model_type_str.replace(' ', '_').lower()}_shap_summary_{class_label.replace(' ', '_')}.png"), bbox_inches='tight')
                plt.close()
        elif shap_values is not None: # Binary classification or regression
            plt.figure()
            shap.summary_plot(shap_values, X_sample_df_for_plot, feature_names=feature_names_list_str, show=False, plot_type="dot")
            plt.title(f"SHAP Summary Plot - {model_type_str}")
            plt.savefig(os.path.join(plot_dir, f"{model_type_str.replace(' ', '_').lower()}_shap_summary.png"), bbox_inches='tight')
            plt.close()
        else:
            print(f"SHAP ({model_type_str}): shap_values are None. Skipping summary plot.")
            return # Cannot proceed to dependence plots if shap_values are None

        print(f"SHAP summary plot(s) saved for {model_type_str}.")

        # --- Dependence Plots ---
        if isinstance(shap_values, list): 
            abs_shap_mean_across_classes = np.mean([np.abs(s) for s in shap_values], axis=0)
            if abs_shap_mean_across_classes.ndim == 2: # (n_samples, n_features)
                mean_abs_shap_per_feature = np.mean(abs_shap_mean_across_classes, axis=0)
            else: # Should not happen if shap_values[i] are (n_samples, n_features)
                print(f"SHAP ({model_type_str}) Warning: Unexpected shape for abs_shap_mean_across_classes. Skipping dependence plots.")
                return
        else: # Binary
             if shap_values.ndim == 2: # (n_samples, n_features)
                mean_abs_shap_per_feature = np.abs(shap_values).mean(0)
             else: # Should not happen
                print(f"SHAP ({model_type_str}) Warning: Unexpected shape for binary shap_values. Skipping dependence plots.")
                return
        
        if len(mean_abs_shap_per_feature) != len(feature_names_list_str):
            print(f"SHAP ({model_type_str}) Warning: Mismatch in length of mean_abs_shap_per_feature ({len(mean_abs_shap_per_feature)}) and feature_names_list ({len(feature_names_list_str)}). Skipping dependence plots.")
        else:
            top_feature_indices = np.argsort(mean_abs_shap_per_feature)[::-1][:min(3, len(feature_names_list_str))]

            for feature_idx_int in top_feature_indices:
                feature_name_as_string = feature_names_list_str[feature_idx_int]
                
                if isinstance(shap_values, list): # Multi-class
                    for i in range(len(shap_values)):
                        plt.figure() 
                        class_label = class_names_list[i] if class_names_list and i < len(class_names_list) else f"Class_{i}"
                        try:
                            # For dependence_plot, 'features' should be the DataFrame X_sample_df_for_plot
                            # 'feature_names' in dependence_plot is for overriding axis labels if 'features' is numpy
                            shap.dependence_plot(
                                feature_name_as_string, # This is the feature to plot on x-axis
                                shap_values[i],         # SHAP values for this class
                                X_sample_df_for_plot,   # The data (Pandas DataFrame)
                                show=False, 
                                interaction_index=None
                            )
                            plt.title(f"SHAP Dependence: {feature_name_as_string} ({class_label}) - {model_type_str}")
                            plt.savefig(os.path.join(plot_dir, f"{model_type_str.replace(' ', '_').lower()}_shap_dependence_{feature_name_as_string.replace('/', '_').replace(':', '_')}_{class_label.replace(' ', '_')}.png"), bbox_inches='tight')
                        except Exception as e_dep:
                            print(f"  SHAP ({model_type_str}): Error generating dependence plot for {feature_name_as_string}, class {class_label}: {e_dep}")
                        finally:
                            plt.close() 
                else: # Binary
                    plt.figure()
                    try:
                        shap.dependence_plot(
                            feature_name_as_string, 
                            shap_values, 
                            X_sample_df_for_plot,
                            show=False, 
                            interaction_index=None
                        )
                        plt.title(f"SHAP Dependence Plot: {feature_name_as_string} - {model_type_str}")
                        plt.savefig(os.path.join(plot_dir, f"{model_type_str.replace(' ', '_').lower()}_shap_dependence_{feature_name_as_string.replace('/', '_').replace(':', '_')}.png"), bbox_inches='tight')
                    except Exception as e_dep:
                        print(f"  SHAP ({model_type_str}): Error generating dependence plot for {feature_name_as_string}: {e_dep}")
                    finally:
                        plt.close()
            print(f"SHAP dependence plots saved for top features of {model_type_str}.")

    except Exception as e:
        print(f"Overall SHAP Error for {model_type_str}: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        
def plot_roc_curves(y_true, y_pred_probas_dict, class_names_list, num_classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, y_pred_proba in y_pred_probas_dict.items():
        if num_classes == 2:
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                 y_scores = y_pred_proba[:, 1]
            elif y_pred_proba.ndim == 1:
                 y_scores = y_pred_proba
            else:
                print(f"Error: y_pred_proba for binary ROC for {model_name} has unexpected shape {y_pred_proba.shape}")
                continue
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != num_classes:
                print(f"Error: y_pred_proba for multi-class ROC for {model_name} has unexpected shape {y_pred_proba.shape}")
                continue
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, linestyle='--', label=f'{model_name} - Class {class_names_list[i]} (AUC = {roc_auc:.3f})')
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
            roc_auc_micro = auc(fpr_micro, tpr_micro)
            ax.plot(fpr_micro, tpr_micro, label=f'{model_name} - Micro Avg (AUC = {roc_auc_micro:.3f})', color='deeppink', linestyle=':', linewidth=3)

    ax.plot([0, 1], [0, 1], 'k--', label='Chance Level (AUC = 0.5)') # T('Chance Level (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate') # T('False Positive Rate')
    ax.set_ylabel('True Positive Rate') # T('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves') # T('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_precision_recall_curves(y_true, y_pred_probas_dict, class_names_list, num_classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name, y_pred_proba in y_pred_probas_dict.items():
        if num_classes == 2:
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                 y_scores = y_pred_proba[:, 1]
            elif y_pred_proba.ndim == 1:
                 y_scores = y_pred_proba
            else:
                print(f"Error: y_pred_proba for binary PR for {model_name} has unexpected shape {y_pred_proba.shape}")
                continue
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ax.plot(recall, precision, label=f'{model_name}')
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != num_classes:
                print(f"Error: y_pred_proba for multi-class PR for {model_name} has unexpected shape {y_pred_proba.shape}")
                continue
            for i in range(num_classes):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                ax.plot(recall, precision, linestyle='--', label=f'{model_name} - Class {class_names_list[i]}')
            # Micro-average PR curve
            # precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_pred_proba.ravel())
            # ax.plot(recall_micro, precision_micro, label=f'{model_name} - Micro Avg', color='deeppink', linestyle=':', linewidth=3)


    ax.set_xlabel('Recall') # T('Recall')
    ax.set_ylabel('Precision') # T('Precision')
    ax.set_title('Precision-Recall Curves') # T('Precision-Recall Curves')
    ax.legend(loc="best") # Consider loc="lower left" or "center left" for PR curves
    ax.grid(True)
    plt.tight_layout()
    return fig