import streamlit as st

TEXTS = {
    "en": {
        # General App Texts
        "app_icon": "⛏️",
        "app_title": "Geochemical Rock Classifier",
        "main_menu_title": "Main Menu",
        "select_page_label": "Select Page",
        "language_label": "Language",

        # Page Titles (for selectbox and potentially for page headers)
        "page_home": "Home",
        "page_data_analysis": "Exploratory Data Analysis",
        "page_run_prediction": "Run Prediction",
        "page_performance_visualizer": "Performance Visualizer",
        "page_model_insights": "Model Insights",
        "page_help_about": "Help / About",
        

        # Home.py Texts
        "home_title": "Geochemical Rock Classifier", # Can be same as app_title or more specific
        "home_subtitle": "Intelligently analyze geochemical data to accurately predict porphyry ore types.",
        "home_intro": """
            This tool utilizes advanced machine learning models, based on 36 major and trace element
            geochemical data from rock samples, to help you quickly distinguish between
            **Copper-rich (Cu-rich)** and **Gold-rich (Au-rich)** porphyry samples.
        """,
        "home_core_features_header": "Core Features:",
        "home_core_feature_1": "**Upload & Predict**: Easily upload your data, select a model, and get instant classification results.",
        "home_core_feature_2": "**Performance Visualization**: If true labels are provided, intuitively assess model performance on your data.",
        "home_core_feature_3": "**Model Insights**: Understand how the model makes decisions and which elemental features are most critical.",
        "home_quick_start_header": "Quick Start",
        "home_button_run_prediction": "🚀 Run Prediction",
        "home_help_run_prediction": "Upload data and run the model for prediction",
        "home_button_view_performance": "📈 View Performance",
        "home_help_view_performance": "Visualize model performance on labeled data",
        "home_button_model_insights": "💡 Model Insights",
        "home_help_model_insights": "Explore model feature importance and interpretability",
        "home_app_overview_header": "Application Overview",
        "home_app_overview_p1": """
            In geological exploration, rapid and accurate identification of deposit types is crucial
            for subsequent resource assessment and development. This project aims to build an
            end-to-end machine learning workflow to:
        """,
        "home_app_overview_li1": "Automatically classify unseen porphyry samples as **Cu-rich** or **Au-rich**.",
        "home_app_overview_li2": "Interpret the key geochemical signatures controlling the classification.",
        "home_app_overview_li3": "Provide a user-friendly offline prediction tool.",
        "home_app_overview_p2": """
            We utilize a dataset of 445 porphyry samples (298 Cu-rich, 147 Au-rich) with 36
            geochemical elemental features (major elements in wt%, trace elements in ppm)
            for model training and evaluation.
        """,
        "home_further_assistance_info": "For detailed usage instructions, feature definitions, model information, and technical details, please visit the **❓ Help / About** page.",
        "home_image_caption": "Porphyry Deposit Illustration",

        # data_analysis.py
        "data_analysis_title": "Exploratory Data Analysis (EDA)",
        "data_analysis_intro": """
        This section presents key visualizations from the exploratory data analysis performed on the 
        **training dataset**. These plots help in understanding data distributions, relationships 
        between elements, and overall data structure before model training.
        """,
        "data_analysis_training_data_notice": "Note: All visualizations in this section are based on the pre-analyzed training data.",
        "data_analysis_image_missing": "Image '{image_name}' not found in 'assets/eda_plots/'. Please ensure it is correctly placed.",
        "data_analysis_scatter_matrix_header": "1. Pair-wise Scatter Matrix (Key Elements)",
        "data_analysis_scatter_matrix_desc": """
        Shows pairwise relationships for a selection of 10 key elements. 
        Diagonal plots show the distribution of each element.
        """,
        "data_analysis_scatter_matrix_caption": "Pair-wise scatter matrix of 10 key elements from the training data.",
        "data_analysis_corr_heatmap_header": "2. Correlation Heatmap",
        "data_analysis_corr_heatmap_desc": """
        Visualizes the Pearson or Spearman correlation coefficients between all elemental features in the training data. 
        Helps identify highly correlated variables.
        """,
        "data_analysis_corr_heatmap_caption": "Correlation heatmap of elemental features from the training data.",
        "data_analysis_pca_biplot_header": "3. PCA Bi-plot (PC1 vs PC2)",
        "data_analysis_pca_biplot_desc": """
        Principal Component Analysis (PCA) bi-plot showing samples (colored by class: Cu-rich/Au-rich) 
        and original feature vectors in the space of the first two principal components, based on training data.
        """,
        "data_analysis_pca_biplot_caption": "PCA bi-plot (PC1 vs PC2) colored by class, from the training data.",
        "data_analysis_geochem_ratios_header": "4. Classic Geochemical Ratio Diagrams",
        "data_analysis_geochem_ratio1_desc": "**K₂O/Na₂O vs SiO₂ Diagram (Training Data):**",
        "data_analysis_geochem_ratio1_caption": "K₂O/Na₂O vs SiO₂ diagram for training samples.",
        "data_analysis_geochem_ratio2_desc": "**Sr/Y vs Y Diagram (Training Data):**",
        "data_analysis_geochem_ratio2_caption": "Sr/Y vs Y diagram for training samples.",
        "data_analysis_plot_quality_notice": """
        All plots are generated with consistent fonts & color palettes and aim for a resolution of ≥ 300 dpi. 
        Each figure includes a title, axis labels, and a legend where applicable.
        """,


        # run_prediction.py
        "run_pred_title": "Run Prediction on New Data",
        "controls_header": "Controls",
        "run_pred_page_intro": "Upload your geochemical data, select a model, and get predictions for Cu-rich or Au-rich samples.",
        "run_pred_upload_header": "1. Upload Your Data",
        "run_pred_upload_label_sidebar": "Upload an Excel (.xlsx) or CSV (.csv) file:",
        "run_pred_model_header": "2. Select Model",
        "run_pred_model_label_sidebar": "Choose a pre-trained model:",
        "run_pred_error_model_load_sidebar": "Model {model_name} could not be loaded.",
        "run_pred_success_model_ready_sidebar": "{model_name} model ready.",
        "run_pred_awaiting_upload": "Awaiting data file upload in the sidebar...",
        "run_pred_error_loading_data": "Failed to load data from the uploaded file.",
        "run_pred_true_label_header": "3. (Optional) Specify True Label Column",
        "run_pred_true_label_checkbox": "My data includes a column with true labels for performance evaluation.",
        "run_pred_true_label_select": "Select the column containing true labels:",
        "run_pred_true_label_none": "None (Proceed without performance evaluation)",
        "run_pred_true_label_help": "This column will be used for performance metrics if available.", 
        "run_pred_warning_no_label_col": "Could not identify a potential label column.", 
        "run_pred_validation_header": "2. Data Validation & Preprocessing", 
        "run_pred_error_validation_failed": "Validation failed: {message}", 
        "run_pred_success_validation": "{message}", 
        "run_pred_spinner_preprocessing": "Preprocessing data...", 
        "run_pred_success_preprocessing_done": "Data preprocessed successfully for prediction.", 
        "run_pred_expander_processed_data": "View Processed Data (First 5 Rows)", 
        "run_pred_run_prediction_header": "3. Run Prediction", 
        "run_pred_submit_button_main": "Predict using {model_name}", 
        "run_pred_input_data_header": "Uploaded Data Preview (First 5 Rows):",
        "run_pred_no_data_warning": "Please upload a data file to proceed.",
        "run_pred_processing_spinner": "Processing data and making predictions...",
        "results_col_predicted_class": "Predicted Class", 
        "results_col_probability_cu_rich": "Prediction Probability ({class_name_cu_rich})", 
        "run_pred_summary_cu_label": "Predicted {class_name_cu_rich} Samples", 
        "run_pred_summary_au_label": "Predicted {class_name_au_rich} Samples",
        "run_pred_success_prediction_done": "Predictions complete! Results are shown above and can be downloaded.", 
        "run_pred_error_prediction_failed_internal": "Prediction process failed internally. Model might not be compatible or data issue.", 
        "run_pred_error_runtime_prediction": "An error occurred during prediction: {error_message}", 
        "run_pred_error_unexpected": "An unexpected error occurred: {error_message}", 
        "run_pred_error_preprocessing_failed": "Data preprocessing failed. Check data compatibility.", 
        "run_pred_results_header": "Prediction Results",
        "run_pred_download_button": "Download Predictions as CSV",
        "run_pred_summary_header": "Prediction Summary:",
        "run_pred_summary_cu": "Predicted Cu-rich samples:",
        "run_pred_summary_au": "Predicted Au-rich samples:",
        "run_pred_error_preprocessor_load": "Critical error: Preprocessor could not be loaded. Predictions cannot proceed.",
        "run_pred_error_file_processing": "Error processing the uploaded file. Please ensure it's a valid CSV or Excel file.",
        "run_pred_error_missing_columns": "Error: The uploaded data is missing one or more required columns. Please check the 'Help / About' page for the list of required features.",
        "run_pred_error_prediction": "An error occurred during prediction.",
        "run_pred_success": "Prediction successful!",

        # performance_visualizer.py
        "perf_viz_title": "Model Performance Visualizer",
        "perf_viz_no_predictions_warning": "No predictions found. Please run predictions on the 'Run Prediction' page first.",
        "perf_viz_no_true_label_info": "To visualize performance, please ensure you uploaded a file with a 'True Label' column and selected it on the 'Run Prediction' page.",
        "perf_viz_raw_data_missing_error": "Original data with labels is missing from session. Please re-run prediction with true labels specified.",
        "perf_viz_true_label_not_found_error": "The specified true label column '{true_label_col}' was not found in the uploaded data.",
        "perf_viz_class_names_error": "CLASS_NAMES configuration is invalid for binary classification visualization.",
        "perf_viz_label_mapping_error": "Some true labels ({unmapped_labels}) in column '{true_label_col}' could not be mapped. Ensure labels are one of {class_names_list}.",
        "perf_viz_true_label_mapping_key_error": "Error mapping true labels. Ensure labels in column '{true_label_col}' are one of {class_names_list}.",
        "perf_viz_predicted_class_col_missing": "'{col_name}' column not found in prediction results.",
        "perf_viz_probability_col_missing": "'{col_name}' column not found in prediction results.",
        "perf_viz_pred_label_processing_error": "Error processing predicted classes for performance metrics.",
        "perf_viz_metrics_subheader": "Performance Metrics",
        "perf_viz_accuracy_label": "Accuracy",
        "perf_viz_precision_label": "Precision ({class_name})",
        "perf_viz_recall_label": "Recall ({class_name})",
        "perf_viz_f1_label": "F1-Score ({class_name})",
        "perf_viz_visualizations_subheader": "Visualizations",
        "perf_viz_cm_tab": "Confusion Matrix",
        "perf_viz_roc_tab": "ROC Curve",
        "perf_viz_pr_tab": "Precision-Recall Curve",
        "perf_viz_cm_title_markdown": "#### Confusion Matrix",
        "perf_viz_cm_caption": """
            Shows the performance of the classification model.
            Rows represent the actual classes, and columns represent the predicted classes.
            - **{class_name_1}**: Copper-dominant porphyry samples.
            - **{class_name_0}**: Gold-dominant porphyry samples.
        """,
        "perf_viz_roc_title_markdown": "#### ROC Curve",
        "perf_viz_selected_model_placeholder": "Selected Model",
        "perf_viz_roc_caption": """
            The Receiver Operating Characteristic (ROC) curve illustrates the diagnostic ability of the classifier
            as its discrimination threshold is varied. The Area Under the Curve (AUC) measures the
            entire two-dimensional area underneath the entire ROC curve from (0,0) to (1,1).
            A model with 100% accuracy has an AUC of 1.0.
        """,
        "perf_viz_pr_title_markdown": "#### Precision-Recall Curve",
        "perf_viz_pr_caption": """
            The Precision-Recall curve shows the tradeoff between precision and recall for different thresholds.
            A high area under the curve represents both high recall and high precision.
            AP (Average Precision) summarizes this curve.
        """,

        # model_insights.py
        "model_insights_title": "Model Interpretability Insights",
        "model_insights_intro": """
        This section provides insights into what features the models consider important.
        Due to computational constraints in a live app, SHAP plots are typically shown as
        pre-generated static images from the model training phase. Feature importances for
        tree-based models and linear SVM are generated based on the selected pre-trained model.
        """,
        "model_insights_model_select_label": "Select a model to view insights:",
        "model_insights_subheader_for_model": "Insights for: {model_name}",
        "model_insights_fi_header": "#### Feature Importances",
        "model_insights_fi_warning_feature_names": "Could not reliably determine transformed feature names; using original. Error: {error_message}",
        "model_insights_fi_plot_caption": "Bar chart showing the relative importance of features for the selected model. Higher scores indicate greater influence.",
        "model_insights_fi_plot_error": "Feature importances could not be plotted for this model type/configuration.",
        "model_insights_fi_svm_info": "Feature importances for SVM are typically shown for linear kernels via coefficients.",
        "model_insights_fi_unavailable_info": "Feature importances are not directly available or easily extractable for {model_name} in this app.",
        "model_insights_fi_load_warning": "Could not load model or preprocessor for {model_name} to show feature importances.",
        "model_insights_fi_dnn_info": "Feature importances for Deep Neural Networks are complex and often explored using techniques like SHAP, which are presented as static plots here based on training data.",
        "model_insights_shap_summary_header": "#### SHAP Summary Plot (Example from Training)",
        "model_insights_shap_summary_desc": """
        SHAP (SHapley Additive exPlanations) values interpret the impact of each feature on individual predictions.
        The summary plot below is an example from the model's training phase, showing feature importance
        and the distribution of SHAP values for each feature.
        """,
        "model_insights_shap_summary_caption": "SHAP Summary Plot for {model_name} (from training data).",
        "model_insights_shap_image_load_warning": "Could not load SHAP {image_type} image: {error_message}",
        "model_insights_shap_image_not_found": "No pre-generated SHAP {image_type} plot found for {model_name} at '{image_path}'.",
        "model_insights_shap_dependence_header": "#### SHAP Dependence Plots (Examples from Training)",
        "model_insights_shap_dependence_desc": """
        SHAP dependence plots show how a single feature's value affects the SHAP value (and thus the prediction),
        potentially highlighting interactions with another feature. These are examples from the training phase.
        """,
        "model_insights_shap_dependence_caption": "SHAP Dependence Plot for {feature_name} ({model_name}, from training).",
        "model_insights_shap_dependence_config_missing": "Configuration for SHAP dependence plots not available for {model_name}.",
        "model_insights_geological_meaning_header": "Geological Meaning of Influential Variables (Discussion)",
        "model_insights_geological_meaning_desc": """
        *(This section should be filled with your discussion of the 5-10 most influential variables
        and their geological meaning, based on your project's interpretability analysis.)*

        For example:
        * **SiO₂ (Silica):** Higher silica content often correlates with more felsic magmas, which can be associated with certain types of porphyry deposits...
        * **Cu/Au Ratio (if engineered):** This directly informs the classification...
        * **K₂O/Na₂O:** Indicates alkalinity, which plays a role in magma evolution and mineralization...
        * **Sr/Y Ratio:** Can be an indicator of slab melt involvement or crustal thickness...
        """,


        # help_about.py
       "help_about_title": "Help / About This Tool",
        "help_about_usage_header": "Usage Instructions",
        "help_about_usage_content": """
    1.  **Navigate to 'Run Prediction'**: Use the sidebar to go to the prediction page.
    2.  **Upload Data**:
        * Click 'Browse files' to upload your sample data.
        * The file must be in `.csv` or `.xlsx` format.
        * Ensure your data contains the 36 required geochemical features (see 'Feature Descriptions' below). Column names should match.
        * A `sample_data.csv` file is available in the `assets/` directory of the project for reference.
    3.  **Select Model**: Choose one of the pre-trained models (Random Forest, XGBoost, SVM, DNN-Keras) from the sidebar.
    4.  **Run Prediction**:
        * If your data includes a column with true labels (e.g., 'Actual_Class'), you can check the box "My data includes a 'True Label' column..." and select that column. This enables performance visualization.
        * Click the 'Predict using [Selected Model]' button.
    5.  **View Results**: Predictions and probabilities will be displayed in a table. You can download these results as a CSV file.
    6.  **Visualize Performance (Optional)**: If you provided a true label column, navigate to the 'Performance Visualizer' page to see the confusion matrix and ROC curve for the predictions made.
    7.  **Explore Model Insights**: Navigate to 'Model Insights' to see feature importances (for some models) and example SHAP plots (static images from training).
    """,
        "help_about_features_header": "Feature Descriptions (Input Data Requirements)",
        "help_about_features_content": """
    Your input data must contain the following 36 major- and trace-element features:
    * **Major Elements (wt %):** SiO₂, TiO₂, Al₂O₃, TFe₂O₃, MnO, MgO, CaO, Na₂O, K₂O, P₂O₅
    * **Trace Elements (ppm):** Rb, Sr, Y, Zr, Nb, Ba, La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, Th, U
    * Missing values in these features will be imputed with zero before further processing (as per project CLR handling).
    """,
        "help_about_model_info_header": "Model Information",
        "help_about_model_info_content": """
    This tool uses four types of pre-trained machine learning models:
    * **Random Forest:** An ensemble learning method using multiple decision trees.
    * **XGBoost:** A gradient boosting framework, known for high performance.
    * **SVM (Support Vector Machine):** A classifier that finds an optimal hyperplane to separate classes.
    * **DNN-Keras:** A Deep Neural Network built using the Keras API.
    Each model was trained on the "2025-Project-Data.xlsx" dataset.
    """,
        "help_about_tech_stack_header": "Library Versions & Tech Stack",
        "help_about_tech_stack_intro": "This application was built using Python and relies on several key libraries:",
        "help_about_tech_stack_caption": "To get exact versions, you can typically use `pip freeze > requirements.txt` in your project's virtual environment and list key ones here.",
        "help_about_project_ack_header": "Project & Acknowledgements",
        "help_about_project_ack_content": """
    * This tool was developed as part of the COMMON TOOLS FOR DATA SCIENCE final project.
    * **Team:** [Your Team Name or Group Number: e.g., 2025GXX-ProjectName]
        * [Team Member 1]
        * [Team Member 2]
        * [Team Member 3]
        * [Team Member 4 (and 5 if applicable)]
    * **Dataset:** "2025-Project-Data.xlsx" (provided for the course).
    * **Inspiration for GUI Structure:** `rascore.streamlit.app` by Mitch Parker.
    """,
        "help_about_contact_info": "For issues or questions, please refer to the project documentation or contact the development team.",
    
    # core/data_handler.py messages
        "data_handler_unsupported_file_type": "Unsupported file type. Please upload a CSV or Excel file.",
        "data_handler_error_loading_data": "Error loading data: {error_message}",
        "data_handler_validation_no_data": "No data loaded for validation.",
        "data_handler_validation_missing_cols": "Missing required columns: {missing_cols_list}. Please ensure your file contains all 36 features.",
        "data_handler_validation_non_numeric": "Column '{column_name}' contains non-numeric data that could not be converted. Please clean your data.",
        "data_handler_validation_success": "Data validation successful. Features ready for preprocessing.",
        "data_handler_error_preprocessing": "Error during data preprocessing: {error_message}",

    # core/model_loader.py messages
        "model_loader_file_not_found": "Model file for {model_name} not found at {file_path}. Please ensure models are correctly placed.",
        "model_loader_dnn_not_implemented": "DNN-Keras model loading is not yet fully implemented in model_loader.py.",
        "model_loader_invalid_model_name": "Invalid model name selected: {model_name}.",
        "model_loader_success_model_load": "{model_name} model loaded successfully.", 
        "model_loader_error_loading_model": "Error loading {model_name} model: {error_message}",
        "model_loader_preprocessor_not_found": "Preprocessor file not found at {file_path}. This is critical for predictions.",
        "model_loader_success_preprocessor_load": "Preprocessor loaded successfully.", 
        "model_loader_error_loading_preprocessor": "Error loading preprocessor: {error_message}。",

    # core/predictor.py messages
        "predictor_unknown_model_type": "Unknown model type for prediction: {model_name}.",
        "predictor_dnn_invalid_model_type": "DNN-Keras model is not a valid Keras model instance.", 

    # core/visualizer.py messages (plot labels and titles)
        "viz_font_warning": "Chinese characters in plots might not display correctly. Ensure a suitable font (e.g., SimHei) is installed.",
        "viz_cm_title": "Confusion Matrix",
        "viz_cm_xlabel": "Predicted Label",
        "viz_cm_ylabel": "True Label",
        "viz_roc_label": "ROC curve (AUC = {auc_score})",
        "viz_roc_xlabel": "False Positive Rate",
        "viz_roc_ylabel": "True Positive Rate",
        "viz_roc_title": "Receiver Operating Characteristic (ROC) - {model_name}",
        "viz_pr_label": "PR curve (AP = {ap_score})",
        "viz_pr_xlabel": "Recall",
        "viz_pr_ylabel": "Precision",
        "viz_pr_title": "Precision-Recall Curve - {model_name}",
        "viz_fi_title": "Top {n_features} Feature Importances - {model_name}",
        "viz_fi_xlabel": "Importance Score",
        "viz_fi_ylabel": "Features"
    },



    "zh": {
        # General App Texts
        "app_icon": "⛏️",
        "app_title": "地球化学岩石分类器",
        "main_menu_title": "主菜单",
        "select_page_label": "选择页面",
        "language_label": "语言",

        # Page Titles
        "page_home": "首页",
        "page_data_analysis": "探索性数据分析",
        "page_run_prediction": "执行预测",
        "page_performance_visualizer": "性能可视化",
        "page_model_insights": "模型洞察",
        "page_help_about": "帮助/关于",

        # Home.py Texts
        "home_title": "地球化学岩石分类器",
        "home_subtitle": "智能分析地球化学数据，精准预测斑岩矿石类型。",
        "home_intro": """
            本工具利用先进的机器学习模型，根据岩石样本的36种主量和微量元素地球化学数据，
            帮助您快速区分**富铜 (Cu-rich)** 和 **富金 (Au-rich)** 斑岩样品。
        """,
        "home_core_features_header": "核心功能:",
        "home_core_feature_1": "**上传与预测**: 轻松上传您的数据，选择模型并获取即时分类结果。",
        "home_core_feature_2": "**性能可视化**: 若提供真实标签，可直观评估模型在您数据上的表现。",
        "home_core_feature_3": "**模型洞察**: 了解模型如何做出决策，哪些元素特征最为关键。",
        "home_quick_start_header": "快速开始",
        "home_button_run_prediction": "🚀 执行预测",
        "home_help_run_prediction": "上传数据并运行模型进行预测",
        "home_button_view_performance": "📈 查看性能",
        "home_help_view_performance": "可视化模型在带标签数据上的性能",
        "home_button_model_insights": "💡 模型洞察",
        "home_help_model_insights": "探索模型特征重要性和可解释性",
        "home_app_overview_header": "应用简介",
        "home_app_overview_p1": """
            地质勘探中，快速准确地识别矿床类型对于后续的资源评估和开发至关重要。
            本项目旨在构建一个端到端的机器学习工作流程，实现：
        """,
        "home_app_overview_li1": "对未见过的斑岩样本进行 **富铜 (Cu-rich)** 或 **富金 (Au-rich)** 的自动分类。",
        "home_app_overview_li2": "解读控制分类结果的关键地球化学特征。",
        "home_app_overview_li3": "提供一个用户友好的离线预测工具。",
        "home_app_overview_p2": """
            我们利用包含445个斑岩样本（298个富铜，147个富金）和36个地球化学元素（主量元素wt%，微量元素ppm）的数据集进行模型训练与评估。
        """,
        "home_further_assistance_info": "有关详细的使用说明、特征定义、模型信息和技术细节，请访问 **❓ Help / About** 页面。",
        "home_image_caption": "斑岩矿床图示",

        # data_analysis.py
        "data_analysis_title": "探索性数据分析 (EDA)",
        "data_analysis_intro": """
        本部分展示了对**训练数据集**进行的探索性数据分析中的关键可视化图表。
        这些图表有助于在模型训练前了解数据分布、元素间的关系以及整体数据结构。
        """,
        "data_analysis_training_data_notice": "注意：本部分所有可视化均基于预先分析的训练数据。",
        "data_analysis_image_missing": "图片 '{image_name}' 在 'assets/eda_plots/' 中未找到。请确保其已正确放置。",
        "data_analysis_scatter_matrix_header": "1. 成对散点矩阵 (关键元素)",
        "data_analysis_scatter_matrix_desc": """
        显示10个关键元素的成对关系。对角线图显示每个元素的分布。
        """,
        "data_analysis_scatter_matrix_caption": "训练数据中10个关键元素的成对散点矩阵。",
        "data_analysis_corr_heatmap_header": "2. 相关性热图",
        "data_analysis_corr_heatmap_desc": """
        可视化训练数据中所有元素特征之间的皮尔逊或斯皮尔曼相关系数。
        有助于识别高度相关的变量。
        """,
        "data_analysis_corr_heatmap_caption": "训练数据中元素特征的相关性热图。",
        "data_analysis_pca_biplot_header": "3. PCA 双标图 (PC1 vs PC2)",
        "data_analysis_pca_biplot_desc": """
        主成分分析 (PCA) 双标图，显示样本（按类别着色：富铜/富金）和原始特征向量
        在前两个主成分空间中的分布，基于训练数据。
        """,
        "data_analysis_pca_biplot_caption": "训练数据中按类别着色的 PCA 双标图 (PC1 vs PC2)。",
        "data_analysis_geochem_ratios_header": "4. 经典地球化学比率图",
        "data_analysis_geochem_ratio1_desc": "**K₂O/Na₂O vs SiO₂ 图 (训练数据):**",
        "data_analysis_geochem_ratio1_caption": "训练样本的 K₂O/Na₂O vs SiO₂ 图。",
        "data_analysis_geochem_ratio2_desc": "**Sr/Y vs Y 图 (训练数据):**",
        "data_analysis_geochem_ratio2_caption": "训练样本的 Sr/Y vs Y 图。",
        "data_analysis_plot_quality_notice": "所有图表均采用一致的字体和调色板生成，并力求达到 ≥ 300 dpi 的分辨率。每个图都包含标题、轴标签和图例（如适用）。",

        # run_prediction.py
        "run_pred_title": "对新数据运行预测",
        "controls_header": "控制面板",
        "run_pred_page_intro": "上传您的地球化学数据，选择模型，并获取富铜或富金样本的预测结果。",
        "run_pred_upload_header": "1. 上传您的数据",
        "run_pred_upload_label_sidebar": "上传 Excel (.xlsx) 或 CSV (.csv) 文件:",
        "run_pred_model_header": "2. 选择模型",
        "run_pred_model_label_sidebar": "选择一个预训练模型:",
        "run_pred_error_model_load_sidebar": "模型 {model_name} 无法加载。",
        "run_pred_success_model_ready_sidebar": "{model_name} 模型已准备就绪。",
        "run_pred_awaiting_upload": "等待在侧边栏上传数据文件...",
        "run_pred_error_loading_data": "加载上传文件时出错。",
        "run_pred_true_label_header": "3. (可选) 指定真实标签列",
        "run_pred_true_label_checkbox": "我的数据包含用于性能评估的真实标签列。",
        "run_pred_true_label_select": "选择包含真实标签的列:",
        "run_pred_true_label_none": "无 (不进行性能评估)",
        "run_pred_true_label_help": "如果可用，此列将用于性能指标评估。", 
        "run_pred_warning_no_label_col": "无法在上传的数据中识别出潜在的标签列。", 
        "run_pred_validation_header": "2. 数据验证与预处理", 
        "run_pred_error_validation_failed": "验证失败: {message}", 
        "run_pred_success_validation": "{message}", 
        "run_pred_spinner_preprocessing": "正在预处理数据...", 
        "run_pred_success_preprocessing_done": "数据已成功预处理以供预测。", 
        "run_pred_expander_processed_data": "查看已处理数据 (前5行)", 
        "run_pred_run_prediction_header": "3. 执行预测", 
        "run_pred_submit_button_main": "使用 {model_name} 进行预测", 
        "run_pred_input_data_header": "已上传数据预览 (前5行):",
        "run_pred_no_data_warning": "请上传数据文件以继续。",
        "run_pred_processing_spinner": "正在处理数据并进行预测...",
        "results_col_predicted_class": "预测类别", 
        "results_col_probability_cu_rich": "预测概率 ({class_name_cu_rich})", 
        "run_pred_summary_cu_label": "预测为 {class_name_cu_rich} 的样本数", 
        "run_pred_summary_au_label": "预测为 {class_name_au_rich} 的样本数", 
        "run_pred_success_prediction_done": "预测完成！结果如上所示，并可供下载。", 
        "run_pred_error_prediction_failed_internal": "预测过程内部失败。模型可能不兼容或数据存在问题。", 
        "run_pred_error_runtime_prediction": "预测过程中发生错误: {error_message}", 
        "run_pred_error_unexpected": "发生意外错误: {error_message}", 
        "run_pred_error_preprocessing_failed": "数据预处理失败。请检查数据兼容性。", 
        "run_pred_results_header": "预测结果",
        "run_pred_download_button": "下载预测结果为 CSV",
        "run_pred_summary_header": "预测摘要:",
        "run_pred_summary_cu": "预测富铜样本数:",
        "run_pred_summary_au": "预测富金样本数:",
        "run_pred_error_preprocessor_load": "关键错误: 预处理器无法加载。无法进行预测。",
        "run_pred_error_file_processing": "处理上传文件时出错。请确保是有效的 CSV 或 Excel 文件。",
        "run_pred_error_missing_columns": "错误：上传的数据缺少一个或多个必需的列。请查看“帮助/关于”页面了解所需的特征列表。",
        "run_pred_error_prediction": "预测过程中发生错误。",
        "run_pred_success": "预测成功！",
        "run_pred_success_intro": "预测已成功完成。以下是您的数据的预测结果。您可以下载结果文件以进行进一步分析。",



        # performance_visualizer.py
       "perf_viz_title": "模型性能可视化工具",
        "perf_viz_no_predictions_warning": "未找到预测结果。请先在“执行预测”页面运行预测。",
        "perf_viz_no_true_label_info": "要可视化性能，请确保您上传的文件包含“真实标签”列，并在“执行预测”页面选择了该列。",
        "perf_viz_raw_data_missing_error": "会话中缺少带标签的原始数据。请重新运行预测并指定真实标签。",
        "perf_viz_true_label_not_found_error": "在上传的数据中未找到指定的真实标签列 '{true_label_col}'。",
        "perf_viz_class_names_error": "CLASS_NAMES 配置对于二分类可视化无效。",
        "perf_viz_label_mapping_error": "列 '{true_label_col}' 中的某些真实标签 ({unmapped_labels}) 无法映射。请确保标签为 {class_names_list} 中的一个。",
        "perf_viz_true_label_mapping_key_error": "映射真实标签时出错。请确保列 '{true_label_col}' 中的标签为 {class_names_list} 中的一个。",
        "perf_viz_predicted_class_col_missing": "在预测结果中未找到 '{col_name}' 列。",
        "perf_viz_probability_col_missing": "在预测结果中未找到 '{col_name}' 列。",
        "perf_viz_pred_label_processing_error": "处理用于性能指标的预测类别时出错。",
        "perf_viz_metrics_subheader": "性能指标",
        "perf_viz_accuracy_label": "准确率",
        "perf_viz_precision_label": "精确率 ({class_name})",
        "perf_viz_recall_label": "召回率 ({class_name})",
        "perf_viz_f1_label": "F1分数 ({class_name})",
        "perf_viz_visualizations_subheader": "可视化图表",
        "perf_viz_cm_tab": "混淆矩阵",
        "perf_viz_roc_tab": "ROC曲线",
        "perf_viz_pr_tab": "精确率-召回率曲线",
        "perf_viz_cm_title_markdown": "#### 混淆矩阵",
        "perf_viz_cm_caption": """
            显示分类模型的性能。
            行代表实际类别，列代表预测类别。
            - **{class_name_1}**: 富铜斑岩样品。
            - **{class_name_0}**: 富金斑岩样品。
        """,
        "perf_viz_roc_title_markdown": "#### ROC曲线",
        "perf_viz_selected_model_placeholder": "所选模型",
        "perf_viz_roc_caption": """
            接收者操作特征（ROC）曲线说明了分类器在其判别阈值变化时的诊断能力。
            曲线下面积（AUC）衡量的是从（0,0）到（1,1）整个ROC曲线下方的整个二维区域。
            一个100%准确的模型其AUC为1.0。
        """,
        "perf_viz_pr_title_markdown": "#### 精确率-召回率曲线",
        "perf_viz_pr_caption": """
            精确率-召回率曲线显示了不同阈值下精确率和召回率之间的权衡。
            曲线下较大的面积代表高精确率和高召回率。
            AP（平均精确率）概括了此曲线。
        """,

        # model_insights.py
        "model_insights_title": "模型可解释性洞察",
        "model_insights_intro": """
        本部分提供了关于模型认为哪些特征重要的见解。
        由于实时应用中的计算限制，SHAP图通常显示为模型训练阶段预先生成的静态图像。
        基于树的模型和线性SVM的特征重要性是根据所选的预训练模型生成的。
        """,
        "model_insights_model_select_label": "选择一个模型以查看洞察:",
        "model_insights_subheader_for_model": "针对模型: {model_name} 的洞察",
        "model_insights_fi_header": "#### 特征重要性",
        "model_insights_fi_warning_feature_names": "无法可靠地确定转换后的特征名称；使用原始名称。错误: {error_message}",
        "model_insights_fi_plot_caption": "条形图显示了所选模型特征的相对重要性。得分越高表示影响越大。",
        "model_insights_fi_plot_error": "无法为此模型类型/配置绘制特征重要性图。",
        "model_insights_fi_svm_info": "SVM的特征重要性通常通过线性核的系数来显示。",
        "model_insights_fi_unavailable_info": "此应用中无法直接获取或轻易提取 {model_name} 的特征重要性。",
        "model_insights_fi_load_warning": "无法加载模型或预处理器以显示 {model_name} 的特征重要性。",
        "model_insights_fi_dnn_info": "深度神经网络的特征重要性较为复杂，通常使用SHAP等技术进行探索，此处显示的是基于训练数据的静态图。",
        "model_insights_shap_summary_header": "#### SHAP 摘要图 (训练阶段示例)",
        "model_insights_shap_summary_desc": """
        SHAP (SHapley Additive exPlanations) 值解释了每个特征对单个预测的影响。
        下面的摘要图是模型训练阶段的一个示例，显示了特征重要性以及每个特征的SHAP值分布。
        """,
        "model_insights_shap_summary_caption": "{model_name} 的SHAP摘要图 (来自训练数据)。",
        "model_insights_shap_image_load_warning": "无法加载SHAP {image_type} 图像: {error_message}",
        "model_insights_shap_image_not_found": "在 '{image_path}' 未找到 {model_name} 的预生成SHAP {image_type} 图。",
        "model_insights_shap_dependence_header": "#### SHAP 依赖图 (训练阶段示例)",
        "model_insights_shap_dependence_desc": """
        SHAP依赖图显示单个特征值如何影响SHAP值（从而影响预测），
        可能突出显示与其他特征的交互作用。这些是训练阶段的示例。
        """,
        "model_insights_shap_dependence_caption": "{feature_name} ({model_name}) 的SHAP依赖图 (来自训练数据)。",
        "model_insights_shap_dependence_config_missing": "{model_name} 的SHAP依赖图配置不可用。",
        "model_insights_geological_meaning_header": "重要变量的地质意义 (讨论)",
        "model_insights_geological_meaning_desc": """
        *（本部分应填写您对5-10个最具影响力的变量及其地质意义的讨论，
        基于您项目的可解释性分析。）*

        例如:
        * **SiO₂ (二氧化硅):**较高的二氧化硅含量通常与更长英质的岩浆相关，这可能与某些类型的斑岩矿床有关...
        * **Cu/Au 比值 (如果经过特征工程):** 这直接为分类提供了信息...
        * **K₂O/Na₂O:** 指示碱度，在岩浆演化和成矿作用中起作用...
        * **Sr/Y 比值:** 可以是板片熔融参与或地壳厚度的指标...
        """,

        # help_about.py
        "help_about_title": "帮助 / 关于此工具",
        "help_about_usage_header": "使用说明",
        "help_about_usage_content": """
    1.  **导航至“执行预测”**：使用侧边栏转到预测页面。
    2.  **上传数据**：
        * 点击“浏览文件”上传您的样本数据。
        * 文件必须为 `.csv` 或 `.xlsx` 格式。
        * 确保您的数据包含36个必需的地球化学特征（参见下方的“特征描述”）。列名应匹配。
        * 项目的 `assets/` 目录中提供了一个 `sample_data.csv` 文件以供参考。
    3.  **选择模型**：从侧边栏选择一个预训练模型（随机森林、XGBoost、支持向量机、DNN-Keras）。
    4.  **运行预测**：
        * 如果您的数据包含带有真实标签的列（例如，“Actual_Class”），您可以勾选“我的数据包含‘真实标签’列...”复选框并选择该列。这将启用性能可视化。
        * 点击“使用[所选模型]进行预测”按钮。
    5.  **查看结果**：预测和概率将显示在表格中。您可以将这些结果下载为CSV文件。
    6.  **可视化性能（可选）**：如果您提供了真实标签列，请导航至“性能可视化工具”页面查看所做预测的混淆矩阵和ROC曲线。
    7.  **探索模型洞察**：导航至“模型洞察”页面查看特征重要性（某些模型）和示例SHAP图（来自训练的静态图像）。
    """,
        "help_about_features_header": "特征描述 (输入数据要求)",
        "help_about_features_content": """
    您的输入数据必须包含以下36个主量和微量元素特征：
    * **主量元素 (wt %):** SiO₂, TiO₂, Al₂O₃, TFe₂O₃, MnO, MgO, CaO, Na₂O, K₂O, P₂O₅
    * **微量元素 (ppm):** Rb, Sr, Y, Zr, Nb, Ba, La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, Th, U
    * 这些特征中的缺失值将在进一步处理前用零填充（根据项目CLR处理方式）。
    """,
        "help_about_model_info_header": "模型信息",
        "help_about_model_info_content": """
    此工具使用四种类型的预训练机器学习模型：
    * **随机森林:** 一种使用多个决策树的集成学习方法。
    * **XGBoost:** 一种梯度提升框架，以高性能著称。
    * **SVM (支持向量机):** 一种通过找到最优超平面来分离类别的分类器。
    * **DNN-Keras:** 一个使用Keras API构建的深度神经网络。
    每个模型都使用 "2025-Project-Data.xlsx" 数据集进行训练。
    """,
        "help_about_tech_stack_header": "库版本和技术栈",
        "help_about_tech_stack_intro": "此应用程序使用Python构建，并依赖于几个关键库：",
        "help_about_tech_stack_caption": "要获取确切版本，您通常可以在项目的虚拟环境中使用 `pip freeze > requirements.txt` 并在此处列出关键版本。",
        "help_about_project_ack_header": "项目与致谢",
        "help_about_project_ack_content": """
    * 此工具是作为“数据科学通用工具”最终项目的一部分开发的。
    * **团队:** [您的团队名称或组号: 例如, 2025GXX-项目名称]
        * [团队成员 1]
        * [团队成员 2]
        * [团队成员 3]
        * [团队成员 4 (及5，如适用)]
    * **数据集:** "2025-Project-Data.xlsx" (课程提供)。
    * **GUI结构灵感来源:** Mitch Parker 的 `rascore.streamlit.app`。
    """,
        "help_about_contact_info": "如有问题或疑问，请参阅项目文档或联系开发团队。",

    # core/data_handler.py messages
        "data_handler_unsupported_file_type": "不支持的文件类型。请上传CSV或Excel文件。",
        "data_handler_error_loading_data": "加载数据时出错: {error_message}",
        "data_handler_validation_no_data": "没有加载用于验证的数据。",
        "data_handler_validation_missing_cols": "缺少必需的列: {missing_cols_list}。请确保您的文件包含所有36个特征。",
        "data_handler_validation_non_numeric": "列 '{column_name}' 包含无法转换的非数字数据。请清理您的数据。",
        "data_handler_validation_success": "数据验证成功。特征已准备好进行预处理。",
        "data_handler_error_preprocessing": "数据预处理过程中出错: {error_message}",

    # core/model_loader.py messages
        "model_loader_file_not_found": "{model_name} 模型文件在 {file_path} 未找到。请确保模型已正确放置。",
        "model_loader_dnn_not_implemented": "DNN-Keras 模型加载尚未在 model_loader.py 中完全实现。",
        "model_loader_invalid_model_name": "所选模型名称无效: {model_name}。",
        "model_loader_success_model_load": "{model_name} 模型加载成功。",
        "model_loader_error_loading_model": "加载 {model_name} 模型时出错: {error_message}",
        "model_loader_preprocessor_not_found": "预处理器文件在 {file_path} 未找到。这对于预测至关重要。",
        "model_loader_success_preprocessor_load": "预处理器加载成功。",
        "model_loader_error_loading_preprocessor": "加载预处理器时出错: {error_message}",

    # core/predictor.py messages
        "predictor_unknown_model_type": "未知的预测模型类型: {model_name}。",
        "predictor_dnn_invalid_model_type": "DNN-Keras 模型不是一个有效的 Keras 模型实例。",

    # core/visualizer.py messages (plot labels and titles)
        "viz_font_warning": "绘图中的中文字符可能无法正确显示。请确保已安装合适的字体（例如：黑体）。",
        "viz_cm_title": "混淆矩阵",
        "viz_cm_xlabel": "预测标签",
        "viz_cm_ylabel": "真实标签",
        "viz_roc_label": "ROC曲线 (AUC = {auc_score})",
        "viz_roc_xlabel": "假阳性率",
        "viz_roc_ylabel": "真阳性率",
        "viz_roc_title": "受试者工作特征曲线 (ROC) - {model_name}",
        "viz_pr_label": "PR曲线 (AP = {ap_score})",
        "viz_pr_xlabel": "召回率",
        "viz_pr_ylabel": "精确率",
        "viz_pr_title": "精确率-召回率曲线 - {model_name}",
        "viz_fi_title": "前 {n_features} 个特征重要性 - {model_name}",
        "viz_fi_xlabel": "重要性得分",
        "viz_fi_ylabel": "特征"
    }
}


# Helper function to get translated text
def T(key, **kwargs):
    """
    Retrieves a translated string for the given key and current language.
    kwargs can be used for simple string formatting.
    Example: T("welcome_user", name="John") where TEXTS has "welcome_user": "Welcome, {name}!"
    """
    lang = st.session_state.get("lang", "en") # Default to English if lang not set
    base_string = TEXTS.get(lang, {}).get(key, f"[{key}]_{lang}")
    if kwargs:
        try:
            return base_string.format(**kwargs)
        except KeyError: # In case a placeholder is in the string but not in kwargs
            return base_string # Return the raw string with placeholders
    return base_string