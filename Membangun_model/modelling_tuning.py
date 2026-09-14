import os
import dagshub
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def main():
    # Inisialisasi DagsHub MLflow Tracking Online
    REPO_OWNER = "rudywijayaa"
    REPO_NAME = "Eksperimen_SML_Rudy-Wijaya"
    
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Hyperparameter_Tuning_Churn")

    # Load Data Preprocessed
    data_dir = "churn_preprocessing"
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # Hyperparameter Tuning dengan GridSearchCV
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    y_pred = best_model.predict(X_test)

    # Hitung Metriks
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # MLflow Manual Logging
    with mlflow.start_run(run_name="RF_Tuning_DagsHub"):
        # Log Hyperparameters
        for param, val in best_params.items():
            mlflow.log_param(param, val)
        mlflow.log_param("cv_folds", 3)

        # Log Metrics Manual
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log Model
        mlflow.sklearn.log_model(best_model, "model")

        # --- Confusion Matrix Plot ---
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - RF Tuned')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        
        mlflow.log_artifact(cm_path) 

        # --- Feature Importance Plot ---
        plt.figure(figsize=(8, 5))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X_train.columns

        sns.barplot(x=importances[indices][:10], y=features[indices][:10], palette="viridis")
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Importance')
        fi_path = "feature_importance.png"
        plt.tight_layout()
        plt.savefig(fi_path)
        plt.close()

        mlflow.log_artifact(fi_path) 

        # Bersihkan file gambar lokal sementara
        if os.path.exists(cm_path): os.remove(cm_path)
        if os.path.exists(fi_path): os.remove(fi_path)

        print("[SUCCESS] Training & Manual Logging ke DagsHub Berhasil!")

if __name__ == "__main__":
    main()