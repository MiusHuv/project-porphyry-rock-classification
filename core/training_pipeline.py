import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt

from core.data_handler import load_and_prepare_data
from core.visualizer import (
    plot_feature_importances, 
    plot_confusion_matrix_heatmap, 
    plot_roc_curves, 
    plot_precision_recall_curves,
    generate_and_save_shap_plots
)
from util.eda_plot import (
    plot_pair_scatter_matrix,
    plot_correlation_heatmap,
    plot_pca_biplot,
    plot_k2o_na2o_vs_sio2,
    plot_sr_y_vs_y
)
from core.train_tree_model import train_random_forest
from core.train_svm_model import train_svm
from core.train_dnn_model import train_pytorch_dnn, SimpleDNN, DEVICE # Import SimpleDNN and DEVICE for prediction
from core.train_xgboost_model import train_xgboost
import torch # Add torch import

from skbio.stats.composition import clr 
from sklearn.preprocessing import StandardScaler

OUTPUT_PLOT_DIR = "output_plots"
if not os.path.exists(OUTPUT_PLOT_DIR):
    os.makedirs(OUTPUT_PLOT_DIR)

# Function to save figures
def save_figure(fig, filename_prefix, plot_dir=OUTPUT_PLOT_DIR):
    if fig is None:
        return
    try:
        filepath = os.path.join(plot_dir, f"{filename_prefix.replace(' ', '_').lower()}.png")
        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")
        plt.close(fig) # Important to close the figure to free memory
    except Exception as e:
        print(f"Error saving figure {filename_prefix}: {e}")
        plt.close(fig) # Still try to close


def create_dummy_data_if_not_exists(filepath, target_col_name, all_expected_cols):
    """Creates a dummy CSV if the specified filepath doesn't exist, including expected columns."""
    if not os.path.exists(filepath):
        print(f"WARNING: Data file '{filepath}' not found.")
        print(f"Creating a dummy CSV for demonstration: 'data/dummy_data.csv' with columns based on image.")
        
        data_dir = os.path.dirname(filepath)
        if not os.path.exists(data_dir) and data_dir:
            os.makedirs(data_dir)
        
        num_samples = 150
        
        # Create DataFrame with all expected columns
        dummy_df_data = {}
        # Add Label and Deposit
        possible_labels = ['Cu-rich PCDs', 'Barren PCDs', 'Intermediate PCDs'] # Example labels
        dummy_df_data[target_col_name] = np.random.choice(possible_labels, size=num_samples)
        if 'Deposit' in all_expected_cols:
            possible_deposits = ['Aktogai Cu', 'Deposit B', 'Deposit C'] # Example deposits
            dummy_df_data['Deposit'] = np.random.choice(possible_deposits, size=num_samples)
        
        # Add numerical features (oxides and trace elements)
        numerical_cols_from_image = [
            'SiO2', 'TiO2', 'Al2O3', 'TFe2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 
            'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Th', 'U'
        ]
        for col in numerical_cols_from_image:
            if col in all_expected_cols:
                # Simulate some plausible data ranges
                if 'SiO2' in col: dummy_df_data[col] = np.random.uniform(40, 80, num_samples)
                elif 'O' in col : dummy_df_data[col] = np.random.uniform(0, 20, num_samples) # Other oxides
                else: dummy_df_data[col] = np.random.uniform(0, 1000, num_samples) # Trace elements
        
        dummy_df = pd.DataFrame(dummy_df_data)

        # Ensure all expected columns are present, fill any missing with 0 for simplicity in dummy
        for col in all_expected_cols:
            if col not in dummy_df.columns and col != target_col_name : # target is already there
                 dummy_df[col] = 0


        # Add some NaNs to test imputation
        for col in dummy_df.columns:
            if col != target_col_name and np.random.rand() > 0.8: # 20% chance to add NaNs
                 dummy_df.loc[dummy_df.sample(frac=0.1).index, col] = np.nan
        
        actual_dummy_filepath = os.path.join(data_dir, "dummy_data.csv")
        dummy_df.to_csv(actual_dummy_filepath, index=False)
        print(f"Created '{actual_dummy_filepath}'.")
        print(f"IMPORTANT: For actual analysis, replace this with your real data file at '{filepath}' "
              f"and ensure TARGET_COLUMN is correctly set to '{target_col_name}'.")
        return actual_dummy_filepath
    return filepath


def run_training_pipeline():
    # --- Configuration ---
    TARGET_COLUMN = "Label"  # Updated based on image
    DATA_FILE_PATH = "data/2025-Project-Data(ESM Table 1).csv"
    
    # Define all expected columns based on the image for dummy data generation and EDA
    # (Order might not be identical to image but names should match)
    ALL_EXPECTED_COLS_FROM_IMAGE = [
        'Label', 'Deposit', 'SiO2', 'TiO2', 'Al2O3', 'TFe2O3', 'MnO', 'MgO', 'CaO', 
        'Na2O', 'K2O', 'P2O5', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 
        'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
        'Hf', 'Ta', 'Th', 'U'
    ]

    # For Geochemical Ratio Plots - Column names from your CSV
    GEOCHEM_K2O_COL = "K2O" 
    GEOCHEM_NA2O_COL = "Na2O"
    GEOCHEM_SIO2_COL = "SiO2"
    GEOCHEM_SR_COL = "Sr"  
    GEOCHEM_Y_COL = "Y"   

    RANDOM_STATE = 42
    
    actual_data_file_path = create_dummy_data_if_not_exists(DATA_FILE_PATH, TARGET_COLUMN, ALL_EXPECTED_COLS_FROM_IMAGE)
    if "dummy_data.csv" in actual_data_file_path:
        print("\nWARNING: USING DUMMY DATA. RESULTS WILL NOT BE MEANINGFUL FOR YOUR PROJECT.\n")
        # If dummy data is used, ensure geochem plot cols exist, if not, they will be skipped by plot functions
        # No need to change GEOCHEM_..._COL here, as plot functions check for column existence.

    # --- 1. Load and Preprocess Data ---
    print("\n--- Stage 1: Data Loading and Preprocessing ---")
    prepared_data = load_and_prepare_data(
        actual_data_file_path, 
        TARGET_COLUMN, 
        test_size=0.25, 
        random_state=RANDOM_STATE,
        apply_outlier_capping_main=True, # Enable outlier capping
        cols_for_log_transform_main=None # No other specific cols for log transform for now
    )
    
    if prepared_data is None:
        print("Pipeline halted due to data loading/preparation errors.")
        return
        
    (X_train, X_test, y_train, y_test, 
     feature_names, class_names, num_classes, 
     label_encoder, scaler, X_processed_for_eda, df_original_for_ratio_plots) = prepared_data

    # --- 2. Exploratory Data Analysis ---
    print("\n--- Stage 2: Exploratory Data Analysis ---")
    
    # For EDA plots that need CLR-transformed data (scatter matrix, correlation, PCA):
    # X_processed_for_eda is X after CLR, log (if any), and OHE, but before scaling.
    # Add the original target label back for hue.
    if TARGET_COLUMN in df_original_for_ratio_plots.columns:
        # Use original target for hue where possible for better interpretability
        X_processed_for_eda['EDA_Target_Hue'] = df_original_for_ratio_plots[TARGET_COLUMN].loc[X_processed_for_eda.index]
    else: # Fallback
        # This alignment is tricky if indices don't match perfectly after processing
        # It's safer to use the encoded y if direct mapping of original target is complex
        # For simplicity, assuming index alignment or using encoded y.
        # y_full_encoded = label_encoder.transform(df_original_for_ratio_plots[TARGET_COLUMN])
        # X_processed_for_eda['EDA_Target_Hue_Encoded'] = y_full_encoded # if using encoded
        pass


    # Key elements for scatter matrix should now be the transformed names (_clr, _log)
    key_elements_for_scatter_clr = [col for col in X_processed_for_eda.columns if '_clr' in col][:10] # Focus on CLR
    if not key_elements_for_scatter_clr and feature_names:
         key_elements_for_scatter_clr = [col for col in feature_names if '_log' in col or 'Deposit' in col][:10]


    eda_scatter_fig = plot_pair_scatter_matrix(X_processed_for_eda, 
                                               key_elements=key_elements_for_scatter_clr, 
                                               target_column_name='EDA_Target_Hue' if 'EDA_Target_Hue' in X_processed_for_eda else None)
    save_figure(eda_scatter_fig, "eda_pair_scatter_matrix_after_clr")

    eda_corr_fig = plot_correlation_heatmap(X_processed_for_eda.select_dtypes(include=np.number), method='pearson')
    save_figure(eda_corr_fig, "eda_correlation_heatmap_after_clr")

    # For PCA: Use X_processed_for_eda (CLR/log/OHE, but unscaled), then scale it for PCA.
    X_pca_candidate_df = X_processed_for_eda.select_dtypes(include=np.number)
    if not X_pca_candidate_df.empty:
        pca_feature_names_viz = X_pca_candidate_df.columns.tolist()
        X_pca_input_scaled = StandardScaler().fit_transform(X_pca_candidate_df)
        
        # Get y_labels corresponding to X_processed_for_eda (full dataset before split)
        y_full_encoded_for_pca = label_encoder.transform(df_original_for_ratio_plots.loc[X_processed_for_eda.index, TARGET_COLUMN])

        pca_fig = plot_pca_biplot(X_pca_input_scaled, y_full_encoded_for_pca, class_names, pca_feature_names_viz)
        save_figure(pca_fig, "eda_pca_biplot_on_clr_data")
    else:
        print("Skipping PCA plot as no numeric features were found in X_processed_for_eda.")

    # Ratio plots use df_original_for_ratio_plots
    k2o_ratio_fig = plot_k2o_na2o_vs_sio2(df_original_for_ratio_plots, GEOCHEM_K2O_COL, GEOCHEM_NA2O_COL, GEOCHEM_SIO2_COL, target_col=TARGET_COLUMN)
    save_figure(k2o_ratio_fig, "eda_k2o_na2o_vs_sio2_ratio")

    # --- 3. Train and Evaluate Models ---
    # (Model training and evaluation section remains largely the same as previous version)
    print("\n--- Stage 3: Model Training and Evaluation ---")
    models_predictions = {} 

    print("\n-- Training Random Forest --")
    rf_model, rf_importances = train_random_forest(
        X_train.copy(), y_train.copy(), feature_names, 
        random_state=RANDOM_STATE, plot_curves=True
    )
    if rf_model:
        y_pred_rf = rf_model.predict(X_test)
        y_pred_proba_rf = rf_model.predict_proba(X_test)
        models_predictions['Random Forest'] = y_pred_proba_rf
        print("\nRandom Forest - Test Set Evaluation:")
        print(classification_report(y_test, y_pred_rf, target_names=class_names, zero_division=0))
        plot_confusion_matrix_heatmap(y_test, y_pred_rf, class_names, "Random Forest")
        rf_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_rf, class_names, "Random Forest")
        save_figure(rf_cm_fig, "rf_confusion_matrix")

        if rf_importances is not None and rf_importances.any():
            rf_fi_fig = plot_feature_importances(rf_importances, feature_names, "Random Forest")
            save_figure(rf_fi_fig, "rf_feature_importances")

    print("\n-- Training SVM --")
    svm_model, svm_importances = train_svm(
        X_train.copy(), y_train.copy(), feature_names, random_state=RANDOM_STATE
    )
    if svm_model:
        y_pred_svm = svm_model.predict(X_test)
        y_pred_proba_svm = svm_model.predict_proba(X_test)
        models_predictions['SVM'] = y_pred_proba_svm
        print("\nSVM - Test Set Evaluation:")
        print(classification_report(y_test, y_pred_svm, target_names=class_names, zero_division=0))
        plot_confusion_matrix_heatmap(y_test, y_pred_svm, class_names, "SVM")
        svm_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_svm, class_names, "SVM")
        save_figure(svm_cm_fig, "svm_confusion_matrix")
        # SVM feature importances are typically not directly available, but if using permutation importance:
        if hasattr(svm_model, 'feature_importances_'):
            svm_importances = svm_model.feature_importances_
        else:
            svm_importances = None
        if svm_importances is not None and svm_importances.any():
            svm_fi_fig = plot_feature_importances(svm_importances, feature_names, "SVM (Permutation)")
            save_figure(svm_fi_fig, "svm_feature_importances")  

    # Model 3: Deep Neural Network (PyTorch)
    print("\n-- Training DNN (PyTorch with Optuna) --")
    if len(X_train) > 50 and len(np.unique(y_train)) > 1:
        X_train_dnn, X_val_dnn, y_train_dnn, y_val_dnn = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )
    else: 
        X_train_dnn, y_train_dnn = X_train, y_train
        X_val_dnn, y_val_dnn = X_test, y_test 
        print("Warning: Using test set as validation for DNN due to small training/stratification constraints.")

    input_dim_dnn = X_train_dnn.shape[1]

    # Ensure data is numpy before passing to PyTorch function if it's pandas
    if isinstance(X_train_dnn, pd.DataFrame): X_train_dnn_np = X_train_dnn.values
    else: X_train_dnn_np = X_train_dnn
    if isinstance(y_train_dnn, pd.Series): y_train_dnn_np = y_train_dnn.values
    else: y_train_dnn_np = y_train_dnn
    if isinstance(X_val_dnn, pd.DataFrame): X_val_dnn_np = X_val_dnn.values
    else: X_val_dnn_np = X_val_dnn
    if isinstance(y_val_dnn, pd.Series): y_val_dnn_np = y_val_dnn.values
    else: y_val_dnn_np = y_val_dnn

    dnn_model_pytorch = train_pytorch_dnn(
        X_train_dnn_np, y_train_dnn_np, X_val_dnn_np, y_val_dnn_np,
        input_dim=input_dim_dnn, 
        num_classes=num_classes,
        epochs=30, # Epochs for Optuna trials and final training phase
        n_optuna_trials=15, # Number of Optuna trials (adjust based on time)
        project_name=f'{TARGET_COLUMN.lower().replace(" ", "_").replace("-","_")}_pytorch_dnn_tuning'
    )

    if dnn_model_pytorch:
        dnn_model_pytorch.eval() # Set model to evaluation mode
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test, dtype=torch.float32).to(DEVICE)
            outputs_test = dnn_model_pytorch(X_test_tensor)
            
            if num_classes == 2:
                # BCEWithLogitsLoss means raw logits, apply sigmoid for probabilities/predictions
                y_pred_proba_dnn_raw = torch.sigmoid(outputs_test).cpu().numpy().flatten()
                y_pred_dnn = (y_pred_proba_dnn_raw > 0.5).astype(int)
                models_predictions['DNN (PyTorch)'] = y_pred_proba_dnn_raw # Probabilities for positive class
            else:
                # CrossEntropyLoss means raw logits, apply softmax for probabilities
                y_pred_proba_dnn_raw = torch.softmax(outputs_test, dim=1).cpu().numpy()
                y_pred_dnn = np.argmax(y_pred_proba_dnn_raw, axis=1)
                models_predictions['DNN (PyTorch)'] = y_pred_proba_dnn_raw # All class probabilities
            
        print("\nDNN (PyTorch) - Test Set Evaluation:")
        print(classification_report(y_test, y_pred_dnn, target_names=class_names, zero_division=0))
        dnn_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_dnn, class_names, "DNN (PyTorch)")
        save_figure(dnn_cm_fig, "dnn_pytorch_confusion_matrix")
        print("DNN Feature importances (e.g., SHAP, Captum) are typically more complex for PyTorch and not plotted by default.")


    # Model 4: XGBoost (New)
    print("\n-- Training XGBoost --")
    xgb_model, xgb_importances = train_xgboost(
        X_train.copy(), y_train.copy(), num_classes=num_classes, feature_names=feature_names,
        random_state=RANDOM_STATE
    )
    if xgb_model:
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)
        models_predictions['XGBoost'] = y_pred_proba_xgb # Add to combined plots
        
        print("\nXGBoost - Test Set Evaluation:")
        print(classification_report(y_test, y_pred_xgb, target_names=class_names, zero_division=0))
        
        xgb_cm_fig = plot_confusion_matrix_heatmap(y_test, y_pred_xgb, class_names, "XGBoost")
        save_figure(xgb_cm_fig, "xgb_confusion_matrix")
        
        if xgb_importances is not None and xgb_importances.any():
            xgb_fi_fig = plot_feature_importances(xgb_importances, feature_names, "XGBoost")
            save_figure(xgb_fi_fig, "xgb_feature_importances")

    if models_predictions:
        print("\n--- Plotting Combined ROC and Precision-Recall Curves ---")
        roc_fig = plot_roc_curves(y_test, models_predictions, class_names, num_classes)
        save_figure(roc_fig, "combined_roc_curves")
        
        pr_fig = plot_precision_recall_curves(y_test, models_predictions, class_names, num_classes)
        save_figure(pr_fig, "combined_precision_recall_curves")
            
    print("\n--- Training and Evaluation Pipeline Complete ---")

    # --- 5. SHAP Value Analysis (after all models are trained) ---
    # X_test is already a DataFrame from data_handler, with final_feature_names
    # X_test was scaled. SHAP explainers generally prefer unscaled or appropriately scaled data
    # depending on the model. For Tree models, original scale is often fine.
    # For KernelExplainer, the scale should match what the model.predict_proba expects.
    # Let's use X_test (scaled) for now, as models were trained on scaled data.
    # If TreeExplainer has issues with scaled data, one might consider using X_test before scaling,
    # but then the SHAP values are on a different scale than model coefficients/splits.
    # For consistency, using the data the model saw (X_test, which is scaled).

    if X_test.empty:
        print("X_test is empty, skipping SHAP plot generation.")
    else:
        sample_size_shap = min(100, len(X_test)) 
        if sample_size_shap > 0:
            # Ensure X_test_sample_shap is a DataFrame with correct feature names
            # X_test should already be a DataFrame with `final_feature_names`
            if not isinstance(X_test, pd.DataFrame): # Should not happen if data_handler is correct
                 X_test_df_for_shap_call = pd.DataFrame(X_test, columns=feature_names)
            else:
                 X_test_df_for_shap_call = X_test

            X_test_sample_shap = X_test_df_for_shap_call.sample(sample_size_shap, random_state=RANDOM_STATE)

            if 'rf_model' in locals() and rf_model:
                generate_and_save_shap_plots(rf_model, X_test_sample_shap, feature_names, 
                                             "Random Forest", num_classes, class_names_list=class_names)
            if 'xgb_model' in locals() and xgb_model:
                generate_and_save_shap_plots(xgb_model, X_test_sample_shap, feature_names, 
                                             "XGBoost", num_classes, class_names_list=class_names)
            
            # SHAP for SVM (KernelExplainer)
            if 'svm_model' in locals() and svm_model:
                print("\nNote: SHAP for SVM (KernelExplainer) can be very slow and is run on a potentially smaller sample.")
                # KernelExplainer can be sensitive to the number of features. 
                # If >20-30 features, it becomes extremely slow.
                # X_test_sample_svm_shap = X_test_sample_shap[final_feature_names[:20]] if len(final_feature_names) > 20 else X_test_sample_shap
                generate_and_save_shap_plots(svm_model, X_test_sample_shap, feature_names, 
                                             "SVM", num_classes, class_names_list=class_names)

            # SHAP for PyTorch DNNs (more complex, requires specific SHAP explainers like DeepExplainer or GradientExplainer)
            if 'dnn_model_pytorch' in locals() and dnn_model_pytorch:
                print("\nSkipping SHAP for PyTorch DNN in this iteration (requires specific SHAP explainers like DeepExplainer/GradientExplainer and careful handling of tensor inputs).")
                # Placeholder for future PyTorch SHAP integration:
                # from core.shap_pytorch_explainer import generate_pytorch_shap_plots # You would create this utility
                # generate_pytorch_shap_plots(dnn_model_pytorch, X_test_sample_shap, final_feature_names, "PyTorch DNN", num_classes, class_names)
        else:
            print("Not enough samples in X_test for SHAP analysis.")

if __name__ == '__main__':
    print("Starting the Porphyry Rock Classification Training Pipeline...")
    run_training_pipeline()