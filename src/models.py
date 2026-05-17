import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import MODELS_DIR, RANDOM_STATE

# ----------------- Machine Learning Models -----------------

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    # Using fewer estimators and max_depth to keep training fast for the pipeline demonstration
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

# ----------------- Deep Learning Models (PyTorch) -----------------

class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> We'll treat our tabular data as seq_len=1 for simplicity
        # or reshape in training
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take last time step
        return self.sigmoid(out)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, input_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_pytorch_model(model, X_train, y_train, epochs=10, batch_size=256, is_lstm=False, is_autoencoder=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Autoencoder trains on normal data only (y==0) usually, but for simplicity here we just show the structure
    # A true anomaly detection autoencoder would be trained on X_train[y_train==0].
    if is_autoencoder:
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0] # not used actually
        
    X_tensor = torch.tensor(X_train.values if hasattr(X_train, 'values') else X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.float32).unsqueeze(1)
    
    if is_lstm:
        # Add sequence dimension
        X_tensor = X_tensor.unsqueeze(1)
        
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if is_autoencoder:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            if is_autoencoder:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x) # Reconstruct input
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")
        
    return model

def predict_pytorch(model, X_test, is_lstm=False, is_autoencoder=False, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_tensor = torch.tensor(X_test.values if hasattr(X_test, 'values') else X_test, dtype=torch.float32).to(device)
    
    if is_lstm:
        X_tensor = X_tensor.unsqueeze(1)
        
    with torch.no_grad():
        outputs = model(X_tensor)
        
        if is_autoencoder:
            # For autoencoder, error is the prediction score
            mse = torch.mean(torch.pow(X_tensor - outputs, 2), dim=1)
            # Find a threshold or just return mse
            # Simplified: if mse > some_value, it's fraud
            # We'll just return the continuous mse score to be used in ROC curves
            return mse.cpu().numpy()
        else:
            probs = outputs.cpu().numpy()
            return probs
