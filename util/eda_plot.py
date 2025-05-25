import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# --- 0. Configuration ---
# Define the path to your data and where to save plots
data_file_path = Path("./data/2025-Project-Data(ESM Table 1).csv")
output_plot_dir = Path("./assets/eda_plots")
output_plot_dir.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

# Define expected columns (features) based on your project description
# SiO2, TiO2, ..., U (36 features)
# You'll need to list all 36 feature column names accurately here
# For brevity, I'll use placeholders. Replace with actual names.
ELEMENT_COLUMNS = [
    'SiO2', 'TiO2', 'Al2O3', 'TFe2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Th', 'U'
]
TARGET_COLUMN = 'Label' # Or whatever your class label column is named
CLASS_NAMES_MAP = {'Cu-rich PCDs': 'Cu-rich', 'Au-rich PCDs': 'Au-rich'} # For PCA plot legend

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv(data_file_path)
except FileNotFoundError:
    print(f"Error: Data file not found at {data_file_path}")
    exit()

# Basic cleaning: remove leading/trailing whitespace from column names if any
df.columns = df.columns.str.strip()

# Ensure all element columns are numeric, coercing errors to NaN
for col in ELEMENT_COLUMNS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: Expected column '{col}' not found in the CSV.")

# Handle missing values (zero replacement for CLR as per course requirements)
df[ELEMENT_COLUMNS] = df[ELEMENT_COLUMNS].fillna(0)

# Separate features and target
X = df[ELEMENT_COLUMNS]
y = df[TARGET_COLUMN].map(CLASS_NAMES_MAP) if TARGET_COLUMN in df.columns else None

# --- CLR Transformation ---
# Add a small constant to avoid log(0) if zeros are present after fillna(0)
# The constant should be very small, e.g., 1e-9 or the smallest non-zero value in the dataset
# However, the course project states "zero replacement for CLR", implying zeros are handled.
# If CLR requires non-zero inputs, this step needs careful consideration.
# For simplicity, assuming direct application after zero fill, but this might need adjustment.
# A common approach for CLR with zeros is to replace them with a very small detection limit value.
# Let's assume for now that the zero-filling is sufficient or a specific strategy for zeros in CLR is defined.

# Ensure no negative values before log, if any, they should be handled (e.g. set to small positive)
X_clr = X.copy()
if (X_clr <= 0).any().any():
    print("Warning: Zero or negative values found. CLR transformation might produce NaNs or errors.")
    # Replace 0s with a very small positive number if necessary for log
    # This depends on the specific CLR implementation or requirements.
    # For example: X_clr[X_clr <= 0] = 1e-9

# Geometric mean for each sample (row-wise)
gmean = np.exp(np.log(X_clr.replace(0, 1e-9)).mean(axis=1)) # Replace 0 to avoid log(0) issues
X_clr = np.log(X_clr.divide(gmean, axis=0).replace(0, 1e-9)) # Replace 0 again if gmean division results in 0

# --- 2. Generate EDA Plots (after CLR) ---

# Plotting function helper
def save_plot(fig, filename, dpi=300):
    fig.savefig(output_plot_dir / filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_plot_dir / filename}")

# A. Pair-wise scatter-matrix of >= 10 key elements
# Select 10 key elements (choose them based on geological importance or variance)
key_elements_for_scatter = ['SiO2', 'K2O', 'Na2O', 'Sr', 'Y', 'Cu', 'Au', 'Th', 'U', 'Rb'] # Example, ensure these are in ELEMENT_COLUMNS
# Filter X_clr to include only these key elements, and add the target for coloring
scatter_df = X_clr[[col for col in key_elements_for_scatter if col in X_clr.columns]].copy()
if y is not None:
    scatter_df[TARGET_COLUMN] = y

# Check if scatter_df is empty or has too few columns
if scatter_df.shape[1] > (1 if y is not None else 0): # Ensure there are feature columns
    pair_plot_fig = sns.pairplot(scatter_df, hue=TARGET_COLUMN if y is not None and TARGET_COLUMN in scatter_df.columns else None, diag_kind='kde', corner=True)
    pair_plot_fig.fig.suptitle("Pair-wise Scatter Matrix (CLR Transformed)", y=1.02)
    save_plot(pair_plot_fig.fig, "scatter_matrix.png")
else:
    print("Skipping scatter matrix: Not enough key elements found or y is None.")


# B. Correlation heat-map (Pearson / Spearman)
plt.figure(figsize=(18, 15)) # Adjust size as needed
correlation_matrix = X_clr.corr(method='spearman') # Or 'spearman'
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1) # annot=True can be very crowded for 36 features
plt.title("Correlation Heatmap (Spearman, CLR Transformed)")
save_plot(plt.gcf(), "correlation_heatmap.png")


# C. PCA bi-plot (PC1 vs PC2) coloured by class
# Standardize data before PCA (important after CLR)
X_scaled = StandardScaler().fit_transform(X_clr)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
if y is not None:
    pc_df[TARGET_COLUMN] = y

plt.figure(figsize=(10, 8))
if y is not None and TARGET_COLUMN in pc_df.columns:
    sns.scatterplot(x='PC1', y='PC2', hue=TARGET_COLUMN, data=pc_df, palette='viridis', s=50, alpha=0.7)
else:
    sns.scatterplot(x='PC1', y='PC2', data=pc_df, s=50, alpha=0.7)

plt.title('PCA Bi-plot (PC1 vs PC2, CLR Transformed)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')

# Add loadings (arrows for features) - this makes it a biplot
loadings = pca.components_.T * np.sqrt(pca.explained_variance_) # Scale loadings by sqrt of explained variance
num_features_to_show = 32 # Show top N features or all if less
for i in range(min(num_features_to_show, loadings.shape[0])):
    plt.arrow(0, 0, loadings[i, 0]*5, loadings[i, 1]*5, color='r', alpha=0.5, head_width=0.1) # Adjust multiplier for arrow length
    plt.text(loadings[i, 0]*5.5, loadings[i, 1]*5.5, X_clr.columns[i], color='b', ha='center', va='center', fontsize=9) # Adjust multiplier

if y is not None: plt.legend(title=TARGET_COLUMN)
plt.grid(True)
save_plot(plt.gcf(), "pca_biplot.png")


# D. Two classic geochemical ratio diagrams
# These ratios should be calculated from the ORIGINAL data (before CLR), as per typical geochemical practice.
# However, the prompt says "After CLR transformation, generate: ... PCA bi-plot ... Two classic geochemical ratio diagrams"
# This is a bit ambiguous. If ratios are on CLR data, their interpretation changes.
# Assuming ratios on ORIGINAL data for geological meaning, then CLR transform the ratios if needed for consistency,
# OR plot original ratios and note this.
# For simplicity, let's calculate ratios on original data (X) and then plot.

# K₂O/Na₂O vs SiO₂
if 'K2O' in X.columns and 'Na2O' in X.columns and 'SiO2' in X.columns:
    # Avoid division by zero if Na2O can be 0
    ratio_k2o_na2o = X['K2O'] / (X['Na2O'] + 1e-9) # Add small constant to denominator
    sio2_values = X['SiO2']

    plt.figure(figsize=(8, 6))
    if y is not None:
        sns.scatterplot(x=sio2_values, y=ratio_k2o_na2o, hue=y, palette='viridis', s=50, alpha=0.7)
        plt.legend(title=TARGET_COLUMN)
    else:
        sns.scatterplot(x=sio2_values, y=ratio_k2o_na2o, s=50, alpha=0.7)
    plt.xlabel("SiO₂ (wt %)")
    plt.ylabel("K₂O/Na₂O")
    plt.title("K₂O/Na₂O vs SiO₂")
    plt.grid(True)
    # plt.ylim(0, ratio_k2o_na2o.quantile(0.99) * 1.1) # Optional: set y-limit to exclude extreme outliers
    # plt.xlim(sio2_values.min()*0.9, sio2_values.quantile(0.99) * 1.1) # Optional
    save_plot(plt.gcf(), "k2o_na2o_vs_sio2.png")
else:
    print("Skipping K2O/Na2O vs SiO2 plot: Required columns not found.")

# Sr/Y vs Y
if 'Sr' in X.columns and 'Y' in X.columns:
    # Avoid division by zero if Y can be 0 for the ratio
    ratio_sr_y = X['Sr'] / (X['Y'] + 1e-9) # Add small constant
    y_values = X['Y']

    plt.figure(figsize=(8, 6))
    if y is not None:
        sns.scatterplot(x=y_values, y=ratio_sr_y, hue=y, palette='viridis', s=50, alpha=0.7)
        plt.legend(title=TARGET_COLUMN)
    else:
        sns.scatterplot(x=y_values, y=ratio_sr_y, s=50, alpha=0.7)
    plt.xlabel("Y (ppm)")
    plt.ylabel("Sr/Y")
    plt.title("Sr/Y vs Y")
    plt.xscale('log') # Y is often plotted on a log scale
    plt.yscale('log') # Sr/Y ratio is also often plotted on a log scale
    plt.grid(True, which="both", ls="-")
    save_plot(plt.gcf(), "sr_y_vs_y.png")
else:
    print("Skipping Sr/Y vs Y plot: Required columns not found.")

print("EDA plots generation complete.")
