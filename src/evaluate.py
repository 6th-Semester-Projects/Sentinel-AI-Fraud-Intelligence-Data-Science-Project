import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report
import pandas as pd
import numpy as np

# Set extreme beautiful style
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "#121212", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(title, fontsize=16, color='white')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(y_true, y_probs, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='cyan', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", facecolor='#1e1e1e', edgecolor='white')
    plt.tight_layout()
    return plt.gcf()

def plot_precision_recall_curve(y_true, y_probs, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#FF007F', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower left", facecolor='#1e1e1e', edgecolor='white')
    plt.tight_layout()
    return plt.gcf()

def generate_report(y_true, y_pred, model_name="Model"):
    print(f"--- Classification Report for {model_name} ---")
    print(classification_report(y_true, y_pred))

def get_metrics_dict(y_true, y_pred, y_probs=None):
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }
    if y_probs is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_probs)
    return metrics
