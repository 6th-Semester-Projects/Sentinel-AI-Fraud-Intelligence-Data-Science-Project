import json
import os

def create_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filename, 'w') as f:
        json.dump(nb, f, indent=2)
    print(f"Created {filename}")

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [source]}

os.makedirs('notebooks', exist_ok=True)

# ----------------- Notebook 1 -----------------
cells_1 = [
    md_cell("# 💳 Layer 1 & 3: Data Collection and EDA\n\nWelcome to the first layer of the **Credit Card Fraud Detection Pipeline**.\n\nIn this notebook, we load the raw transactions and perform deep Exploratory Data Analysis (EDA) to understand class imbalance and feature distributions."),
    code_cell("import sys\nimport os\nsys.path.append('..')\nfrom src.preprocessing import load_data\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# Set beautiful style\nplt.style.use('dark_background')"),
    md_cell("## 1. Load Data"),
    code_cell("try:\n    df = load_data()\n    print(f'Data Loaded Successfully! Shape: {df.shape}')\nexcept Exception as e:\n    print(e)"),
    md_cell("## 2. Target Class Imbalance"),
    code_cell("if 'df' in locals():\n    plt.figure(figsize=(6,4))\n    sns.countplot(data=df, x='Class', palette=['#00e5ff', '#ff007f'])\n    plt.title('Extreme Class Imbalance', fontsize=16)\n    plt.yscale('log')\n    plt.show()")
]
create_notebook('notebooks/01_Data_Collection_and_EDA.ipynb', cells_1)

# ----------------- Notebook 2 -----------------
cells_2 = [
    md_cell("# ⚙️ Layer 2 & 4: Preprocessing and Feature Engineering\n\nHere we apply transformations, scale the features robustly, and use **SMOTE** to handle the extreme class imbalance."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.preprocessing import full_preprocessing_pipeline\n\nprint('Running Full Preprocessing Pipeline...')\nX_train, X_test, y_train, y_test = full_preprocessing_pipeline(apply_smote=True)\nprint(f'Train set shape after SMOTE: {X_train.shape}')")
]
create_notebook('notebooks/02_Preprocessing_and_Feature_Engineering.ipynb', cells_2)

# ----------------- Notebook 3 -----------------
cells_3 = [
    md_cell("# 🤖 Layer 5: Model Training (Machine Learning)\n\nTraining traditional ML models including **Logistic Regression**, **Random Forest**, and **XGBoost**."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.preprocessing import full_preprocessing_pipeline\nfrom src.models import train_logistic_regression, train_random_forest, train_xgboost\n\nX_train, X_test, y_train, y_test = full_preprocessing_pipeline(apply_smote=True)\n\nprint('Training Logistic Regression...')\nlr_model = train_logistic_regression(X_train, y_train)\n\nprint('Training Random Forest...')\nrf_model = train_random_forest(X_train, y_train)\n\nprint('Training XGBoost...')\nxgb_model = train_xgboost(X_train, y_train)\nprint('All ML Models Trained successfully!')")
]
create_notebook('notebooks/03_Model_Training_ML.ipynb', cells_3)

# ----------------- Notebook 4 -----------------
cells_4 = [
    md_cell("# 🧠 Layer 5: Model Training (Deep Learning)\n\nTraining extreme high-level Deep Learning architectures: **DNN**, **LSTM**, and **Autoencoder** using PyTorch."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.preprocessing import full_preprocessing_pipeline\nfrom src.models import DNN, LSTMNetwork, Autoencoder, train_pytorch_model\n\nX_train, X_test, y_train, y_test = full_preprocessing_pipeline(apply_smote=True)\ninput_dim = X_train.shape[1]\n\nprint('Training DNN...')\ndnn = DNN(input_dim)\ndnn = train_pytorch_model(dnn, X_train, y_train, epochs=2)\nprint('DNN Trained!')")
]
create_notebook('notebooks/04_Model_Training_DL.ipynb', cells_4)

# ----------------- Notebook 5 -----------------
cells_5 = [
    md_cell("# 📊 Layer 6: Evaluation and Comparison\n\nFinally, we compare all 6 techniques and visualize the results using beautiful ROC curves and Confusion Matrices."),
    code_cell("import sys\nsys.path.append('..')\nimport pandas as pd\nimport os\nfrom src.config import MODELS_DIR\n\nmetrics_file = os.path.join(MODELS_DIR, 'metrics.csv')\nif os.path.exists(metrics_file):\n    df_metrics = pd.read_csv(metrics_file)\n    display(df_metrics.style.background_gradient(cmap='Blues'))\nelse:\n    print('Please run main.py first to generate the metrics file.')")
]
create_notebook('notebooks/05_Evaluation_and_Comparison.ipynb', cells_5)

print("Notebook generation complete.")
