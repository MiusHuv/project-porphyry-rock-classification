# File: page/data_analysis.py
# -----------------------------------------------------------------------------
import streamlit as st
from util.language import T, TEXTS
from PIL import Image
from pathlib import Path

def show_page():
    st.title(T("data_analysis_title", default="Exploratory Data Analysis (EDA)"))
    
    st.markdown(T("data_analysis_intro", default="""
    This section presents key visualizations from the exploratory data analysis performed on the 
    **training dataset**. These plots help in understanding data distributions, relationships 
    between elements, and overall data structure before model training.
    """))
    st.info(T("data_analysis_training_data_notice", default="Note: All visualizations in this section are based on the pre-analyzed training data."))

    # Define a helper function to display images
    def display_eda_image(image_filename, caption_key, default_caption):
        img_path = Path(__file__).parent.parent / "assets" / "eda_plots" / image_filename
        try:
            img = Image.open(img_path)
            st.image(img, caption=T(caption_key, default=default_caption), use_container_width=True)
        except FileNotFoundError:
            st.warning(T("data_analysis_image_missing", image_name=image_filename, default=f"Image '{image_filename}' not found in 'assets/eda_plots/'."))

    st.markdown("---")

    # 1. Pair-wise scatter-matrix
    st.subheader(T("data_analysis_scatter_matrix_header", default="1. Pair-wise Scatter Matrix (Key Elements)"))
    st.markdown(T("data_analysis_scatter_matrix_desc", default="""
    Shows pairwise relationships for a selection of 10 key elements. 
    Diagonal plots show the distribution of each element.
    """))
    display_eda_image("scatter_matrix.png", "data_analysis_scatter_matrix_caption", "Pair-wise scatter matrix of 10 key elements.")

    st.markdown("---")

    # 2. Correlation heat-map
    st.subheader(T("data_analysis_corr_heatmap_header", default="2. Correlation Heatmap"))
    st.markdown(T("data_analysis_corr_heatmap_desc", default="""
    Visualizes the Pearson or Spearman correlation coefficients between all elemental features. 
    Helps identify highly correlated variables.
    """))
    display_eda_image("correlation_heatmap.png", "data_analysis_corr_heatmap_caption", "Correlation heatmap of elemental features.")

    st.markdown("---")

    # 3. PCA bi-plot
    st.subheader(T("data_analysis_pca_biplot_header", default="3. PCA Bi-plot (PC1 vs PC2)"))
    st.markdown(T("data_analysis_pca_biplot_desc", default="""
    Principal Component Analysis (PCA) bi-plot showing samples (colored by class: Cu-rich/Au-rich) 
    and original feature vectors in the space of the first two principal components.
    """))
    display_eda_image("pca_biplot.png", "data_analysis_pca_biplot_caption", "PCA bi-plot (PC1 vs PC2) colored by class.")

    st.markdown("---")

    # 4. Geochemical ratio diagrams
    st.subheader(T("data_analysis_geochem_ratios_header", default="4. Classic Geochemical Ratio Diagrams"))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(T("data_analysis_geochem_ratio1_desc", default="**K₂O/Na₂O vs SiO₂ Diagram:**"))
        display_eda_image("k2o_na2o_vs_sio2.png", "data_analysis_geochem_ratio1_caption", "K₂O/Na₂O vs SiO₂ diagram.")
    
    with col2:
        st.markdown(T("data_analysis_geochem_ratio2_desc", default="**Sr/Y vs Y Diagram:**"))
        display_eda_image("sr_y_vs_y.png", "data_analysis_geochem_ratio2_caption", "Sr/Y vs Y diagram.")

    st.markdown("---")
    st.markdown(T("data_analysis_plot_quality_notice", default="""
    All plots are generated with consistent fonts & color palettes and aim for a resolution of ≥ 300 dpi. 
    Each figure includes a title, axis labels, and a legend where applicable.
    """))

if __name__ == "__main__":
    if 'lang' not in st.session_state: st.session_state.lang = "en"
    show_page()