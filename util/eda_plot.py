import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# EDA

def plot_pair_scatter_matrix(df, key_elements, target_column_name=None, sample_n=1000):
    if df is None or df.empty:
        print("EDA: DataFrame is empty. Skipping scatter matrix.")
        return None
    if not key_elements:
        print("EDA: No key elements provided for scatter matrix.")
        return None
    
    actual_key_elements = [col for col in key_elements if col in df.columns]
    if len(actual_key_elements) < 2:
        print(f"EDA: Too few valid key elements ({len(actual_key_elements)}) for scatter matrix. Need at least 2.")
        return None

    print(f"EDA: Generating pair-wise scatter matrix for: {actual_key_elements}")
    df_plot = df.copy()
    if len(df_plot) > sample_n:
        df_plot = df_plot.sample(n=sample_n, random_state=42)

    # sns.pairplot returns a PairGrid object, its figure is in g.fig
    if target_column_name and target_column_name in df_plot.columns:
        g = sns.pairplot(df_plot, vars=actual_key_elements, 
                         hue=target_column_name, diag_kind='kde', 
                        #  corner=True, 
                         palette='viridis')
    else:
        if target_column_name:
            print(f"EDA Warning: Target column '{target_column_name}' not found for hue in scatter matrix.")
        g = sns.pairplot(df_plot, vars=actual_key_elements, diag_kind='kde', corner=True)
    
    g.fig.suptitle("Pair-wise Scatter Matrix of Key Elements", y=1.02, fontsize=16) # T("Pair-wise Scatter Matrix of Key Elements")
    # g.fig.set_dpi(1500)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Apply to g.fig if needed after suptitle, or let pairplot handle
    return g.fig

def plot_correlation_heatmap(df, method='pearson'):
    if df is None or df.empty:
        print("EDA: DataFrame is empty. Skipping correlation heatmap.")
        return None
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        print("EDA: No numeric columns found for correlation heatmap.")
        return None

    print(f"EDA: Generating {method} correlation heatmap.")
    fig, ax = plt.subplots(figsize=(14, 12)) # Increased size for better annotation visibility
    correlation_matrix = numeric_df.corr(method=method)
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, annot_kws={"size": 8}) # Smaller annotation font
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax, annot_kws={"size": 8}) # Smaller annotation font
    ax.set_title(f"{method.capitalize()} Correlation Heatmap", fontsize=16) # T(f"{method.capitalize()} Correlation Heatmap")
    plt.tight_layout()
    return fig

def plot_pca_biplot(X_scaled, y_encoded_labels, class_names, feature_names):
    if len(X_scaled) != len(y_encoded_labels):
            raise ValueError(f"Dimension mismatch: X has {len(X_scaled)} samples but y_encoded_labels has {len(y_encoded_labels)} samples")

    if X_scaled is None or X_scaled.shape[0] == 0 or X_scaled.shape[1] < 2 :
        print("EDA: Not enough data or features for PCA biplot. Skipping.")
        return None
        
    print("EDA: Generating PCA bi-plot (PC1 vs PC2).")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pc_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    target_for_hue = "Target_Class" # T("Target_Class")
    if target_for_hue and y_encoded_labels is not None:
        # Ensure y_encoded_labels is 1D
        if hasattr(y_encoded_labels, 'ndim') and y_encoded_labels.ndim > 1:
            if y_encoded_labels.shape[1] == 1:
                y_encoded_labels = y_encoded_labels.ravel()
            else:
                # Handle or raise error for multi-dimensional y_encoded_labels not suitable for hue
                # For now, let's try to use the first column if it's multi-output, or raise more specific error
                print("EDA Warning: y_encoded_labels has more than one column. Using only the first column for hue.")
                y_encoded_labels = y_encoded_labels[:, 0]

        # Safely create labels for the hue
        pc_df[target_for_hue] = [
            class_names[int(label)] if class_names is not None and 0 <= int(label) < len(class_names) 
            else f'Class {int(label)}' 
            for label in y_encoded_labels
        ]
        # Original line causing error:
        # pc_df[target_for_hue] = [class_names[int(label)] if class_names and int(label) < len(class_names) else f'Class {int(label)}' for label in y_encoded_labels]
        # Corrected line:
        # pc_df[target_for_hue] = [class_names[int(label)] if class_names is not None and 0 <= int(label) < len(class_names) else f'Class {int(label)}' for label in y_encoded_labels]
    else:
        target_for_hue = None # No hue if no target info

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue=target_for_hue, ax=ax, palette='viridis', s=50, alpha=0.7)

    ax.set_xlabel('Principal Component 1 (PC1)') # T('Principal Component 1 (PC1)')
    ax.set_ylabel('Principal Component 2 (PC2)') # T('Principal Component 2 (PC2)')
    ax.set_title('PCA Bi-plot (PC1 vs PC2)') # T('PCA Bi-plot (PC1 vs PC2)')
    ax.legend(title='Classes') # T('Classes')
    ax.grid(True)

    if feature_names and len(feature_names) == X_scaled.shape[1]:
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        num_features_to_show = min(len(feature_names), 10)
        loading_magnitudes = np.sqrt(loadings[:,0]**2 + loadings[:,1]**2)
        top_feature_indices = np.argsort(loading_magnitudes)[-num_features_to_show:]

        for i in top_feature_indices:
            ax.arrow(0, 0, loadings[i, 0]*2.5, loadings[i, 1]*2.5, 
                      color='r', alpha=0.6, head_width=0.05, head_length=0.05, zorder=3)
            ax.text(loadings[i, 0]*2.8, loadings[i, 1]*2.8, feature_names[i], 
                     color='black', ha='center', va='center', fontsize=9, zorder=3,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5))
    
    plt.tight_layout()
    print(f"PCA Explained Variance Ratio: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    return fig

def plot_k2o_na2o_vs_sio2(df, k2o_col, na2o_col, sio2_col, target_col=None):
    if not all(col in df.columns for col in [k2o_col, na2o_col, sio2_col]):
        print(f"EDA Warning: One or more columns ({k2o_col}, {na2o_col}, {sio2_col}) not found. Skipping K₂O/Na₂O vs SiO₂ plot.")
        return None
    
    df_plot = df.copy()
    df_plot[na2o_col] = df_plot[na2o_col].replace(0, 1e-9)
    df_plot['K2O_Na2O_Ratio'] = df_plot[k2o_col] / df_plot[na2o_col]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    if target_col and target_col in df_plot.columns:
        sns.scatterplot(data=df_plot, x=sio2_col, y='K2O_Na2O_Ratio', hue=target_col, alpha=0.7, ax=ax, palette='viridis')
        ax.legend(title=target_col)
    else:
        sns.scatterplot(data=df_plot, x=sio2_col, y='K2O_Na2O_Ratio', alpha=0.7, ax=ax)
        
    ax.set_xlabel(f'{sio2_col} (%)') # T(f'{sio2_col} (%)')
    ax.set_ylabel('K₂O/Na₂O Ratio') # T('K₂O/Na₂O Ratio')
    ax.set_title('Geochemical Ratio: K₂O/Na₂O vs SiO₂') # T('Geochemical Ratio: K₂O/Na₂O vs SiO₂')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_sr_y_vs_y(df, sr_col, y_col, target_col=None):
    if not all(col in df.columns for col in [sr_col, y_col]):
        print(f"EDA Warning: One or more columns ({sr_col}, {y_col}) not found. Skipping Sr/Y vs Y plot.")
        return None

    df_plot = df.copy()
    # Ensure Y_col is not zero for division and log scale
    df_plot[y_col] = df_plot[y_col].apply(lambda x: x if x > 1e-9 else 1e-9)
    df_plot['Sr_Y_Ratio'] = df_plot[sr_col] / df_plot[y_col]
    # Ensure Sr_Y_Ratio is positive for log scale
    df_plot['Sr_Y_Ratio'] = df_plot['Sr_Y_Ratio'].apply(lambda x: x if x > 1e-9 else 1e-9)

    fig, ax = plt.subplots(figsize=(10, 7))
    if target_col and target_col in df_plot.columns:
        sns.scatterplot(data=df_plot, x=y_col, y='Sr_Y_Ratio', hue=target_col, alpha=0.7, ax=ax, palette='viridis')
        ax.legend(title=target_col)
    else:
        sns.scatterplot(data=df_plot, x=y_col, y='Sr_Y_Ratio', alpha=0.7, ax=ax)
        
    ax.set_xlabel(f'{y_col} (ppm)') # T(f'{y_col} (ppm)')
    ax.set_ylabel('Sr/Y Ratio') # T('Sr/Y Ratio')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Geochemical Ratio: Sr/Y vs Y') # T('Geochemical Ratio: Sr/Y vs Y')
    ax.grid(True, which="both", ls="-")
    plt.tight_layout()
    return fig