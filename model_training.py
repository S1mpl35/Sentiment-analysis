"""
Module 4: Train baseline (market only) and augmented (market + sentiment) models
Outputs: Accuracy, Precision, Recall, F1 score, confusion matrices, feature importance plot
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Optional: set matplotlib for Chinese characters (if needed)
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculate classification metrics.
    
    Parameters:
        y_true: ground truth labels
        y_pred: predicted labels
        model_name: name of the model (for printing)
    
    Returns:
        dict: metrics including accuracy, precision, recall, f1
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    return metrics

def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    """
    Plot confusion matrix using seaborn heatmap.
    
    Parameters:
        y_true: ground truth labels
        y_pred: predicted labels
        title: plot title
        save_path: if provided, save figure to this path
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    plt.show()

def plot_feature_importance(model, feature_names, top_k=10, save_path=None):
    """
    Plot feature importance bar chart for random forest model.
    
    Parameters:
        model: trained RandomForestClassifier
        feature_names: list of feature names
        top_k: number of top features to display
        save_path: if provided, save figure to this path
    """
    importances = model.feature_importances_
    # Sort indices by importance descending
    indices = np.argsort(importances)[::-1][:top_k]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(indices)), top_importances, align='center')
    plt.yticks(range(len(indices)), top_features)
    plt.gca().invert_yaxis()  # highest importance on top
    plt.xlabel('Feature Importance')
    plt.title(f'Random Forest Augmented Model - Top {top_k} Feature Importances')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")
    plt.show()

def main():
    print("=== Module 4: Model Training & Comparison ===")
    
    # Check if split data file exists
    data_path = "data/split_data.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Split data file not found: {data_path}. Please run Module 3 first.")
    
    # Load pre-split data
    data = np.load(data_path, allow_pickle=True)
    X_train_m = data['X_train_m']
    X_test_m = data['X_test_m']
    X_train_e = data['X_train_e']
    X_test_e = data['X_test_e']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names_enhanced = data['feature_names_enhanced'].tolist()
    
    print(f"Loaded data: training samples={len(X_train_m)}, test samples={len(X_test_m)}")
    print(f"Number of enhanced features: {len(feature_names_enhanced)}")
    
    # ----- Logistic Regression -----
    print("\n" + "="*50)
    print("Training Logistic Regression models...")
    
    # Baseline (market only)
    lr_base = LogisticRegression(max_iter=1000, random_state=42)
    lr_base.fit(X_train_m, y_train)
    y_pred_lr_base = lr_base.predict(X_test_m)
    plot_confusion_matrix(y_test, y_pred_lr_base,
                          "Logistic Regression - Baseline Confusion Matrix",
                          "confusion_lr_base.png")
    
    # Augmented (market + sentiment)
    lr_enh = LogisticRegression(max_iter=1000, random_state=42)
    lr_enh.fit(X_train_e, y_train)
    y_pred_lr_enh = lr_enh.predict(X_test_e)
    plot_confusion_matrix(y_test, y_pred_lr_enh,
                          "Logistic Regression - Augmented Confusion Matrix",
                          "confusion_lr_enh.png")
    
    # ----- Random Forest -----
    print("\n" + "="*50)
    print("Training Random Forest models...")
    
    # Baseline (market only)
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_base.fit(X_train_m, y_train)
    y_pred_rf_base = rf_base.predict(X_test_m)
    plot_confusion_matrix(y_test, y_pred_rf_base,
                          "Random Forest - Baseline Confusion Matrix",
                          "confusion_rf_base.png")
    
    # Augmented (market + sentiment)
    rf_enh = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_enh.fit(X_train_e, y_train)
    y_pred_rf_enh = rf_enh.predict(X_test_e)
    plot_confusion_matrix(y_test, y_pred_rf_enh,
                          "Random Forest - Augmented Confusion Matrix",
                          "confusion_rf_enh.png")
    
    # Feature importance for the best model (random forest augmented)
    plot_feature_importance(rf_enh, feature_names_enhanced, top_k=10,
                            save_path="feature_importance.png")
    
    # ----- Evaluation Summary -----
    models = {
        'LR_Baseline': (y_test, y_pred_lr_base),
        'LR_Enhanced': (y_test, y_pred_lr_enh),
        'RF_Baseline': (y_test, y_pred_rf_base),
        'RF_Enhanced': (y_test, y_pred_rf_enh)
    }
    results = {}
    for name, (yt, yp) in models.items():
        results[name] = evaluate_model(yt, yp, name)
    
    # Print comparison table
    print("\n" + "="*60)
    print("📊 Experiment Results Summary")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12}")
    print("-"*60)
    for name in ['LR_Baseline', 'LR_Enhanced', 'RF_Baseline', 'RF_Enhanced']:
        acc = results[name]['accuracy']
        f1 = results[name]['f1']
        print(f"{name:<25} {acc:.4f}        {f1:.4f}")
    
    # Additional insight: improvement from baseline
    print("\n" + "-"*60)
    print("Performance Improvement from Baseline:")
    lr_improve = results['LR_Enhanced']['f1'] - results['LR_Baseline']['f1']
    rf_improve = results['RF_Enhanced']['f1'] - results['RF_Baseline']['f1']
    print(f"  Logistic Regression: +{lr_improve:.4f} F1")
    print(f"  Random Forest:       +{rf_improve:.4f} F1")
    
    print("\n✅ Module 4 completed")

if __name__ == "__main__":
    main()