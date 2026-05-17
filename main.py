import os
import pandas as pd
from src.config import DATA_FILE, MODELS_DIR
from src.preprocessing import full_preprocessing_pipeline
from src.models import (
    train_logistic_regression, train_random_forest, train_xgboost,
    save_model, DNN, LSTMNetwork, Autoencoder, train_pytorch_model, predict_pytorch
)
from src.evaluate import get_metrics_dict
import warnings
warnings.filterwarnings('ignore')

def run_pipeline():
    print("==================================================")
    print("=== CREDIT CARD FRAUD DETECTION - PIPELINE EXECUTION")
    print("==================================================")
    
    if not DATA_FILE.exists():
        print(f"[ERROR] Dataset not found at {DATA_FILE}")
        print("Please download creditcard.csv from Kaggle and place it in data/raw/")
        return

    # --- Layer 1-4: Data Collection, Preprocessing & Feature Engineering ---
    print("\n[LAYER 1-4] Preprocessing Data & Applying SMOTE...")
    X_train, X_test, y_train, y_test = full_preprocessing_pipeline(apply_smote=True)
    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    
    metrics_list = []
    
    # --- Layer 5: ML Models ---
    print("\n[LAYER 5] Training Machine Learning Models...")
    
    # Logistic Regression
    print(" -> Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    save_model(lr_model, 'logistic_regression.pkl')
    y_pred_lr = lr_model.predict(X_test)
    y_probs_lr = lr_model.predict_proba(X_test)[:, 1]
    metrics_list.append({"Model": "Logistic Regression", **get_metrics_dict(y_test, y_pred_lr, y_probs_lr)})
    
    # Random Forest
    print(" -> Training Random Forest (approx 1-2 mins)...")
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, 'random_forest.pkl')
    y_pred_rf = rf_model.predict(X_test)
    y_probs_rf = rf_model.predict_proba(X_test)[:, 1]
    metrics_list.append({"Model": "Random Forest", **get_metrics_dict(y_test, y_pred_rf, y_probs_rf)})
    
    # XGBoost
    print(" -> Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    save_model(xgb_model, 'xgboost.pkl')
    y_pred_xgb = xgb_model.predict(X_test)
    y_probs_xgb = xgb_model.predict_proba(X_test)[:, 1]
    metrics_list.append({"Model": "XGBoost", **get_metrics_dict(y_test, y_pred_xgb, y_probs_xgb)})
    
    # --- Layer 5: DL Models ---
    print("\n[LAYER 5] Training Deep Learning Models (PyTorch)...")
    input_dim = X_train.shape[1]
    
    # DNN
    print(" -> Training Deep Neural Network (DNN)...")
    dnn_model = DNN(input_dim)
    dnn_model = train_pytorch_model(dnn_model, X_train, y_train, epochs=5)
    torch_save_path = os.path.join(MODELS_DIR, 'dnn.pth')
    import torch
    torch.save(dnn_model.state_dict(), torch_save_path)
    print(f"Model saved to {torch_save_path}")
    
    y_probs_dnn = predict_pytorch(dnn_model, X_test)
    y_pred_dnn = (y_probs_dnn > 0.5).astype(int).flatten()
    metrics_list.append({"Model": "Deep Neural Network", **get_metrics_dict(y_test, y_pred_dnn, y_probs_dnn.flatten())})
    
    # LSTM
    print(" -> Training LSTM Network...")
    lstm_model = LSTMNetwork(input_dim)
    lstm_model = train_pytorch_model(lstm_model, X_train, y_train, epochs=3, is_lstm=True)
    torch.save(lstm_model.state_dict(), os.path.join(MODELS_DIR, 'lstm.pth'))
    
    y_probs_lstm = predict_pytorch(lstm_model, X_test, is_lstm=True)
    y_pred_lstm = (y_probs_lstm > 0.5).astype(int).flatten()
    metrics_list.append({"Model": "LSTM Network", **get_metrics_dict(y_test, y_pred_lstm, y_probs_lstm.flatten())})
    
    # Autoencoder
    print(" -> Training Autoencoder (Anomaly Detection)...")
    ae_model = Autoencoder(input_dim)
    # Train only on normal data
    ae_model = train_pytorch_model(ae_model, X_train, y_train, epochs=5, is_autoencoder=True)
    torch.save(ae_model.state_dict(), os.path.join(MODELS_DIR, 'autoencoder.pth'))
    
    y_scores_ae = predict_pytorch(ae_model, X_test, is_autoencoder=True)
    # Threshold for AE (e.g. 95th percentile of normal data train error, simplified here)
    threshold = pd.Series(y_scores_ae).quantile(0.95)
    y_pred_ae = (y_scores_ae > threshold).astype(int)
    # ROC for AE uses the continuous reconstruction error as probability-like score
    metrics_list.append({"Model": "Autoencoder", **get_metrics_dict(y_test, y_pred_ae, y_scores_ae)})
    
    # --- Layer 6: Evaluation ---
    print("\n[LAYER 6] Saving Evaluation Metrics...")
    metrics_df = pd.DataFrame(metrics_list)
    metrics_csv = os.path.join(MODELS_DIR, 'metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metrics saved to {metrics_csv}")
    
    print("\n==================================================")
    print("[SUCCESS] PIPELINE EXECUTION COMPLETE!")
    print("Now you can run the Streamlit Dashboard:")
    print("   streamlit run dashboard/app.py")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline()
