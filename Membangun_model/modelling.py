import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # Load dataset hasil preprocessing
    data_dir = "churn_preprocessing"
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    # Set MLflow autolog untuk baseline
    mlflow.set_experiment("Baseline_Model_Churn")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Baseline"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    main()