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

def generate_and_save_shap_plots(model, X_sample_df_orig, feature_names_list, model_type_str, 
                                 num_classes_model, class_names_list=None, 
                                 plot_dir="output_plots/shap_plots", 
                                 X_background_df_orig=None, 
                                 device_for_dnn=None): # Pass the device for DNN
    if not SHAP_AVAILABLE: print(f"SHAP ({model_type_str}): Library not available. Skipping."); return
    
    X_sample_df = X_sample_df_orig.copy()
    if X_sample_df.empty: print(f"SHAP ({model_type_str}): X_sample_df is empty. Skipping."); return
    
    try:
        X_sample_df = X_sample_df[feature_names_list] # Ensure column order and selection
    except KeyError as e:
        print(f"SHAP ({model_type_str}) Error: Columns in X_sample_df do not match feature_names_list. {e}"); return
            
    feature_names_list_str = [str(fn) for fn in feature_names_list]
    X_sample_df.columns = feature_names_list_str

    print(f"\nGenerating SHAP plots for {model_type_str}...")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)

    try:
        explainer = None
        shap_values_raw = None

        if model_type_str.lower() in ['random forest', 'xgboost']:
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(X_sample_df, check_additivity=False)
        elif model_type_str.lower() == 'svm' and hasattr(model, 'predict_proba'):
            # ... (SVM KernelExplainer logic as before) ...
            print(f"SHAP ({model_type_str}): Preparing KernelExplainer...")
            # Use a smaller sample for background if X_background_df_orig is not provided for SVM
            background_svm = X_background_df_orig if X_background_df_orig is not None else X_sample_df
            if len(background_svm) > 50: background_svm = shap.sample(background_svm[feature_names_list_str], 50, random_state=42)
            else: background_svm = background_svm[feature_names_list_str]

            def svm_predict_proba_for_shap(data_for_svm_np):
                data_for_svm_df = pd.DataFrame(data_for_svm_np, columns=feature_names_list_str)
                return model.predict_proba(data_for_svm_df)
            explainer = shap.KernelExplainer(svm_predict_proba_for_shap, background_svm)
            print(f"SHAP ({model_type_str}): Calculating SHAP values with KernelExplainer...")
            shap_values_raw = explainer.shap_values(X_sample_df, nsamples='auto', check_additivity=False)
            print(f"SHAP ({model_type_str}): SHAP values calculated.")
        
        elif model_type_str.lower() == 'pytorch dnn':
            if device_for_dnn is None:
                device_for_dnn = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model.eval() 
            model.to(device_for_dnn)

            if X_background_df_orig is None:
                print(f"SHAP ({model_type_str}): No background data provided for DeepExplainer, using a sample from X_sample_df.")
                X_background_df = shap.sample(X_sample_df, min(50, len(X_sample_df)), random_state=42)
            else:
                X_background_df = X_background_df_orig.copy()
                if not isinstance(X_background_df, pd.DataFrame) : X_background_df = pd.DataFrame(X_background_df, columns=feature_names_list_str)
                else: X_background_df = X_background_df[feature_names_list_str] # Ensure column order

            background_tensor = torch.tensor(X_background_df.values, dtype=torch.float32).to(device_for_dnn)
            X_sample_tensor = torch.tensor(X_sample_df.values, dtype=torch.float32).to(device_for_dnn)
            
            print(f"SHAP ({model_type_str}): Preparing DeepExplainer with background shape {background_tensor.shape}...")
            explainer = shap.DeepExplainer(model, background_tensor)
            print(f"SHAP ({model_type_str}): Calculating SHAP values with DeepExplainer...")
            shap_output_from_explainer = explainer.shap_values(X_sample_tensor) 
            print(f"SHAP ({model_type_str}): SHAP values calculated. Type from explainer: {type(shap_output_from_explainer)}")

            # Handle output from DeepExplainer: could be tensor, list of tensors, numpy array, or list of numpy arrays
            if isinstance(shap_output_from_explainer, list):
                # If it's a list, check if elements are tensors or numpy arrays
                if all(torch.is_tensor(s_val) for s_val in shap_output_from_explainer):
                    shap_values_raw = [s_val.cpu().numpy() for s_val in shap_output_from_explainer]
                    print(f"SHAP ({model_type_str}): Converted list of tensors to list of numpy arrays.")
                elif all(isinstance(s_val, np.ndarray) for s_val in shap_output_from_explainer):
                    shap_values_raw = shap_output_from_explainer # Already list of numpy arrays
                    print(f"SHAP ({model_type_str}): Output is already a list of numpy arrays.")
                else:
                    print(f"SHAP ({model_type_str}): Output is a list with mixed types. Cannot process."); return
            elif torch.is_tensor(shap_output_from_explainer): 
                shap_values_raw = shap_output_from_explainer.cpu().numpy()
                print(f"SHAP ({model_type_str}): Converted tensor to numpy array.")
            elif isinstance(shap_output_from_explainer, np.ndarray): # THIS IS THE CASE THAT WAS HIT
                shap_values_raw = shap_output_from_explainer # Already a numpy array
                print(f"SHAP ({model_type_str}): Output is already a numpy array.")
            else:
                print(f"SHAP ({model_type_str}): Unexpected output type from DeepExplainer: {type(shap_output_from_explainer)}"); return

        # --- Standardize shap_values structure for plotting (this logic should remain largely the same) ---
        shap_values_for_plotting = None
        is_multi_class_for_loops = False 

        if isinstance(shap_values_raw, list): 
            if num_classes_model == 2 and len(shap_values_raw) == 2: 
                shap_values_for_plotting = shap_values_raw[1] 
                print(f"SHAP ({model_type_str}): Binary (from list[2]), using SHAP for positive class.")
            elif len(shap_values_raw) == num_classes_model: 
                shap_values_for_plotting = shap_values_raw
                is_multi_class_for_loops = True
                print(f"SHAP ({model_type_str}): Multi-class (from list[{num_classes_model}]).")
            else:
                print(f"SHAP ({model_type_str}) Error: shap_values_raw is list len {len(shap_values_raw)}, num_classes {num_classes_model}. Mismatch."); return
        
        elif isinstance(shap_values_raw, np.ndarray): 
            if num_classes_model == 2: # Binary Classification Cases
                if shap_values_raw.ndim == 3 and shap_values_raw.shape[-1] == 2:
                    # Case: (samples, features, 2 classes) - e.g., some RF/SVM outputs
                    shap_values_for_plotting = shap_values_raw[:, :, 1] 
                    print(f"SHAP ({model_type_str}): Binary (from 3D array with 2 class outputs), using SHAP for positive class (shape: {shap_values_for_plotting.shape}).")
                elif shap_values_raw.ndim == 3 and shap_values_raw.shape[-1] == 1:
                    # Case: (samples, features, 1 class output) - e.g., PyTorch DNN with single output neuron for binary
                    shap_values_for_plotting = np.squeeze(shap_values_raw, axis=-1) # Remove the last dimension
                    # Alternative: shap_values_for_plotting = shap_values_raw[:, :, 0]
                    print(f"SHAP ({model_type_str}): Binary (from 3D array with 1 class output, e.g. DNN), using squeezed SHAP values (shape: {shap_values_for_plotting.shape}).")
                elif shap_values_raw.ndim == 2:
                    # Case: (samples, features) - e.g., XGBoost binary
                    shap_values_for_plotting = shap_values_raw
                    print(f"SHAP ({model_type_str}): Binary (from 2D array, e.g. XGBoost).")
                else:
                    print(f"SHAP ({model_type_str}) Error: Binary case, but ndarray shape {shap_values_raw.shape} not recognized."); return
            
            elif num_classes_model > 2: # Multi-class Classification Cases
                if shap_values_raw.ndim == 3 and shap_values_raw.shape[-1] == num_classes_model:
                    # Case: (samples, features, num_classes) - e.g., some TreeExplainer multi-class outputs
                    temp_list = [shap_values_raw[:, :, i] for i in range(num_classes_model)]
                    shap_values_for_plotting = temp_list
                    is_multi_class_for_loops = True
                    print(f"SHAP ({model_type_str}): Multi-class (from 3D array converted to list).")
                else:
                    print(f"SHAP ({model_type_str}) Error: Multi-class case, but ndarray shape {shap_values_raw.shape} not recognized for num_classes {num_classes_model}."); return
            else: # Should not happen if num_classes_model is always >= 2
                 print(f"SHAP ({model_type_str}) Error: num_classes_model is {num_classes_model}, not handled for ndarray."); return
        else:
            print(f"SHAP ({model_type_str}) Error: Unexpected type for shap_values_raw: {type(shap_values_raw)}"); return

        # --- Summary Plot & Dependence Plots (this logic should mostly work if shap_values_for_plotting is correctly structured) ---
        # ... (The existing summary and dependence plot generation code from the previous response) ...
        # Ensure X_sample_df_for_plot is used, which has string columns
        X_sample_df_for_plot = X_sample_df.copy() # Has string columns from earlier

        if is_multi_class_for_loops and isinstance(shap_values_for_plotting, list): 
            for i in range(len(shap_values_for_plotting)):
                plt.figure() 
                class_label = class_names_list[i] if class_names_list and i < len(class_names_list) else f"Class_{i}"
                shap.summary_plot(shap_values_for_plotting[i], X_sample_df_for_plot, feature_names=feature_names_list_str, show=False, plot_type="dot")
                plt.title(f"SHAP Summary for {class_label} - {model_type_str}")
                save_path = os.path.join(plot_dir, f"{model_type_str.replace(' ', '_').lower()}_shap_summary_{class_label.replace(' ', '_')}.png")
                plt.savefig(save_path, bbox_inches='tight'); plt.close()
                print(f"  Saved: {save_path}")
        elif shap_values_for_plotting is not None and shap_values_for_plotting.ndim == 2: 
            plt.figure()
            shap.summary_plot(shap_values_for_plotting, X_sample_df_for_plot, feature_names=feature_names_list_str, show=False, plot_type="dot")
            plt.title(f"SHAP Summary Plot - {model_type_str}")
            save_path = os.path.join(plot_dir, f"{model_type_str.replace(' ', '_').lower()}_shap_summary.png")
            plt.savefig(save_path, bbox_inches='tight'); plt.close()
            print(f"  Saved: {save_path}")
        else:
            print(f"SHAP ({model_type_str}): shap_values_for_plotting (shape: {shap_values_for_plotting.shape if hasattr(shap_values_for_plotting, 'shape') else 'N/A'}) not suitable for summary. Skipping.");
            if not (is_multi_class_for_loops and isinstance(shap_values_for_plotting, list)): return 

        print(f"SHAP summary plot(s) generated for {model_type_str}.")

        mean_abs_shap_per_feature = None
        if is_multi_class_for_loops and isinstance(shap_values_for_plotting, list):
            abs_shap_mean_across_classes = np.mean([np.abs(s) for s in shap_values_for_plotting], axis=0)
            if abs_shap_mean_across_classes.ndim == 2: mean_abs_shap_per_feature = np.mean(abs_shap_mean_across_classes, axis=0)
        elif shap_values_for_plotting is not None and shap_values_for_plotting.ndim == 2: 
            mean_abs_shap_per_feature = np.abs(shap_values_for_plotting).mean(0)
        
        if mean_abs_shap_per_feature is None:
            print(f"SHAP ({model_type_str}) Warning: Could not calculate mean_abs_shap_per_feature for dependence plots. Skipping.")
            return 

        if len(mean_abs_shap_per_feature) != len(feature_names_list_str):
            print(f"SHAP ({model_type_str}) Warning: Mismatch in length of mean_abs_shap ({len(mean_abs_shap_per_feature)}) and feature_names ({len(feature_names_list_str)}). Skipping dependence plots.")
        else:
            top_feature_indices = np.argsort(mean_abs_shap_per_feature)[::-1][:min(3, len(feature_names_list_str))]
            for feature_idx_int in top_feature_indices:
                feature_name_as_string = feature_names_list_str[feature_idx_int]
                if is_multi_class_for_loops and isinstance(shap_values_for_plotting, list):
                    for i in range(len(shap_values_for_plotting)):
                        plt.figure() 
                        class_label = class_names_list[i] if class_names_list and i < len(class_names_list) else f"Class_{i}"
                        try:
                            shap.dependence_plot(feature_name_as_string, shap_values_for_plotting[i], X_sample_df_for_plot, show=False, interaction_index=None)
                            plt.title(f"SHAP Dependence: {feature_name_as_string} ({class_label}) - {model_type_str}")
                            save_path = os.path.join(plot_dir, f"{model_type_str.replace(' ', '_').lower()}_shap_dependence_{feature_name_as_string.replace('/', '_').replace(':', '_')}_{class_label.replace(' ', '_')}.png")
                            plt.savefig(save_path, bbox_inches='tight'); plt.close()
                            print(f"  Saved: {save_path}")
                        except Exception as e_dep: print(f"  SHAP ({model_type_str}): Error in dependence plot for {feature_name_as_string}, class {class_label}: {e_dep}"); plt.close()
                elif shap_values_for_plotting is not None and shap_values_for_plotting.ndim == 2: 
                    plt.figure()
                    try:
                        shap.dependence_plot(feature_name_as_string, shap_values_for_plotting, X_sample_df_for_plot, show=False, interaction_index=None)
                        plt.title(f"SHAP Dependence Plot: {feature_name_as_string} - {model_type_str}")
                        save_path = os.path.join(plot_dir, f"{model_type_str.replace(' ', '_').lower()}_shap_dependence_{feature_name_as_string.replace('/', '_').replace(':', '_')}.png")
                        plt.savefig(save_path, bbox_inches='tight'); plt.close()
                        print(f"  Saved: {save_path}")
                    except Exception as e_dep: print(f"  SHAP ({model_type_str}): Error in dependence plot for {feature_name_as_string}: {e_dep}"); plt.close()
            print(f"SHAP dependence plots generated for top features of {model_type_str}.")
    except Exception as e:
        print(f"Overall SHAP Error for {model_type_str}: {e}"); import traceback; traceback.print_exc(); plt.close('all')

        
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
