<h1 align="center">
  🛡️ Sentinel AI: Fraud Intelligence Infrastructure
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B.svg?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success.svg" alt="Status">
</p>

<h3 align="center">An enterprise-grade financial security infrastructure built to intercept and analyze credit card cyber-threats in real-time.</h3>

---

## 🌌 Overview
**Sentinel AI** is not just a machine learning model; it is a fully comprehensive, end-to-end Cyber-Security & Fraud Intelligence system. Built to handle massive class-imbalance in financial datasets, the system leverages an ensemble of Machine Learning algorithms alongside Deep Neural Networks to intercept anomalies.

This project was developed for a 6th-semester Data Science course, pushing the boundaries of academic requirements to deliver a Silicon Valley-grade software product.

## ✨ God-Tier Features

* 🤖 **The AI Council (Ensemble Voting):** A multi-model architecture where Logistic Regression, Random Forest, and XGBoost collectively vote to authorize or intercept transactions.
* 🧠 **Explainable AI (XAI):** Integrated `SHAP` framework to provide human-readable logic for every neural decision (e.g., "Feature V14 contributed +35% to risk").
* 🌍 **Live 3D Threat Radar:** An interactive, orthographic 3D globe plotting simulated geographical origins of intercepted cyber-threats.
* 🕸️ **Dark Web Crime Ring Network:** Generates interactive Spider-web graphs using `PyVis` to visualize money laundering connections between compromised accounts and mules.
* 📄 **Forensic PDF Generator:** Automated creation of immutable, professional audit reports for intercepted threats.
* 🧬 **Continuous Learning Simulation:** A "Self-Healing" terminal that simulates real-time backpropagation and weight updates upon encountering mutated threat signatures.
* 🔊 **Voice Alerts:** Real-time Text-to-Speech (TTS) voice warnings when an infiltration attempt is thwarted.

---

## 🏗️ System Architecture (Layer-wise Implementation)

Our system is structured into 6 distinct, highly optimized Data Science layers:

- **Layer 1 (Data Collection):** Processing over 284,807 transactions.
- **Layer 2 (Preprocessing):** Handling null values, logarithmic scaling of monetary amounts, and time-feature engineering.
- **Layer 3 (Exploratory Data Analysis):** High-end statistical plotting, correlation heatmaps, and variance analysis.
- **Layer 4 (Resampling):** Utilizing **SMOTE** (Synthetic Minority Over-sampling Technique) to eradicate the 99.8% safe vs 0.2% fraud class imbalance.
- **Layer 5 (Model Training):** Training robust ML pipelines alongside PyTorch-powered Deep Neural Networks (DNN) and LSTMs.
- **Layer 6 (Evaluation):** Metric extraction using ROC-AUC, Precision-Recall curves, and Confusion Matrices.

---

## 🚀 Installation & Deployment

### 1. Local Setup
Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/6th-Semester-Projects/Sentinel-AI-Fraud-Intelligence-Data-Science-Project.git
cd Sentinel-AI-Fraud-Intelligence-Data-Science-Project

# Install required dependencies
pip install -r requirements.txt

# Launch the Sentinel AI Dashboard
streamlit run dashboard/app.py
```

### 2. Streamlit Community Cloud (Free Deployment)
1. Fork or upload this repository to your GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub and select this repository.
4. Set the main file path to `dashboard/app.py`.
5. Click **Deploy**.

---

## 📊 Dataset Information
The dataset used is a highly confidential credit card transaction ledger where feature confidentiality is maintained via PCA (Principal Component Analysis). Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'.

---

## 👨‍💻 Development Team
**BSCS-F-23-A**
* **Muhammad Maauz Mansoor** (233599)
* **Zain Riaz** (233597)
* **Zahid Zafar** (233579)

---
> *"Predicting the future of cyber-threats, one transaction at a time."*
