# project-2025-porphyry-rock-classification

好的，这是一份基于您项目内容和功能的中英文双语 `README.md` 文档。

---

# 斑岩矿床类型分类器 (Porphyry Rock Classifier)

## English

### Project Description

This project implements an intelligent system for classifying porphyry ore types (Copper-rich vs. Gold-rich) based on geochemical data. It utilizes various machine learning models (Random Forest, SVM, XGBoost, PyTorch DNN) and provides a Streamlit-based Graphical User Interface (GUI) for users to:
* Upload their geochemical data for prediction.
* Visualize model performance if true labels are provided.
* Gain insights into model decisions through feature importance and SHAP analysis.
* Explore pre-computed Exploratory Data Analysis (EDA) results from the training dataset.

The application supports both English and Chinese languages.

### Environment Setup

It is recommended to use a Conda virtual environment.

1.  **Create Conda Environment (Recommended)**:
    ```bash
    conda create -n porphyry_classifier python=3.11
    conda activate porphyry_classifier
    ```

2.  **Install Dependencies**:
    The project dependencies are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include: `streamlit`, `pandas`, `scikit-learn`, `joblib`, `xgboost`, `matplotlib`, `seaborn`, `openpyxl`, `Pillow`, `scikit-bio`, `torch`, and `optuna` (Optuna is primarily for the training phase).

3.  **Development Container (Optional)**:
    This project includes a `devcontainer.json` configuration. If you are using VS Code with Docker and the "Dev Containers" extension, you can open the project in a container, and the environment will be set up automatically according to this configuration.

### Running the Model Training Pipeline

Before running the GUI application for the first time, or if you need to retrain the models, you must execute the training pipeline script. This script will process the data, train all models, evaluate them, and save the necessary artifacts (models, preprocessors, plots).

**Run the script from the project root directory:**
```bash
python core/training_pipeline.py
```

**Purpose and Outputs of the Training Pipeline:**
* Loads and preprocesses data from `data/2025-Project-Data(ESM Table 1).csv`.
* Trains Random Forest, SVM, XGBoost, and PyTorch DNN models.
* Saves trained models to the `models/` directory (e.g., `random_forest_model.joblib`, `pytorch_dnn_model.pth`).
* Saves essential preprocessing objects (scaler, label encoder classes, final feature names) to the `models/` directory.
* Generates and saves EDA plots, model evaluation plots (confusion matrices, ROC/PR curves, feature importance plots) to the `assets/eda_plots/` directory.
* Generates and saves SHAP analysis plots (summary and dependence plots) to the `assets/shap_plots/` directory.

### Running the Streamlit GUI Application

Ensure that the model training pipeline has been run successfully and the `models/` and `assets/` directories contain the necessary files.

**Run the application from the project root directory:**
```bash
streamlit run app.py
```
The application will open in your default web browser.

### Project Structure
* `app.py`: Main entry point for the Streamlit GUI application.
* `page/`: Contains modules for each page/view of the GUI.
* `core/`: Contains core logic for data handling, model training, prediction, and visualization.
* `util/`: Contains utility modules (e.g., language internationalization, EDA plotting functions).
* `models/`: Stores trained models and preprocessing objects.
* `assets/`: Stores static assets like images and pre-generated plots.
* `data/`: Stores raw data files.
* `requirements.txt`: Python package dependencies.
* `.devcontainer/`: Configuration for development containers.

---

## 中文

### 项目描述

本项目实现了一个基于地球化学数据对斑岩矿床类型（富铜型与富金型）进行智能分类的系统。该系统运用了多种机器学习模型（随机森林、支持向量机、XGBoost、PyTorch深度神经网络），并提供了一个基于Streamlit的图形用户界面（GUI），用户可以通过该界面：
* 上传自己的地球化学数据进行预测。
* 如果提供了真实标签，可以可视化模型性能。
* 通过特征重要性和SHAP分析来洞察模型的决策过程。
* 浏览从训练数据集中预先计算生成的探索性数据分析（EDA）结果。

该应用支持中文和英文两种语言。

### 环境设置

推荐使用Conda创建虚拟环境。

1.  **创建Conda环境 (推荐)**:
    ```bash
    conda create -n porphyry_classifier python=3.11
    conda activate porphyry_classifier
    ```

2.  **安装依赖**:
    项目依赖项已在 `requirements.txt` 文件中列出。请使用pip进行安装：
    ```bash
    pip install -r requirements.txt
    ```
    主要依赖包括: `streamlit`, `pandas`, `scikit-learn`, `joblib`, `xgboost`, `matplotlib`, `seaborn`, `openpyxl`, `Pillow`, `scikit-bio`, `torch`, 以及 `optuna` (Optuna主要用于训练阶段)。

3.  **开发容器 (可选)**:
    本项目包含一个 `devcontainer.json` 配置文件。如果您使用VS Code和Docker以及“Dev Containers”扩展，您可以在容器中打开项目，环境将根据此配置自动设置。

### 运行模型训练流程

在首次运行GUI应用程序之前，或者当您需要重新训练模型时，必须执行模型训练流程脚本。该脚本将处理数据、训练所有模型、评估它们，并保存必要的构件（模型、预处理器、图表）。

**请在项目根目录下运行此脚本：**
```bash
python core/training_pipeline.py
```

**训练流程的目的与产出：**
* 加载并预处理 `data/2025-Project-Data(ESM Table 1).csv` 数据。
* 训练随机森林、SVM、XGBoost和PyTorch DNN模型。
* 将训练好的模型保存到 `models/` 目录下（例如 `random_forest_model.joblib`, `pytorch_dnn_model.pth`）。
* 将必要的预处理对象（scaler、标签编码器的类别、最终特征名称）保存到 `models/` 目录下。
* 生成并保存EDA图表、模型评估图表（混淆矩阵、ROC/PR曲线、特征重要性图）到 `assets/eda_plots/` 目录。
* 生成并保存SHAP分析图表（摘要图和依赖图）到 `assets/shap_plots/` 目录。

### 运行Streamlit GUI应用

请确保模型训练流程已成功运行，并且 `models/` 和 `assets/` 目录包含了必要的文件。

**在项目根目录下运行：**
```bash
streamlit run app.py
```
应用程序将在您的默认网页浏览器中打开。

### 项目结构
* `app.py`: Streamlit GUI应用的主入口文件。
* `page/`: 包含GUI各个页面/视图的模块。
* `core/`: 包含核心数据处理、模型训练、预测和可视化逻辑。
* `util/`: 包含通用工具模块（例如，语言国际化、EDA绘图函数）。
* `models/`: 存储训练好的模型和预处理对象。
* `assets/`: 存储图片、预生成的图表等静态资源。
* `data/`: 存储原始数据文件。
* `requirements.txt`: Python包依赖项。
* `.devcontainer/`: 开发容器配置。
