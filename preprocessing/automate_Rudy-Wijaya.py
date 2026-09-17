import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def main():
    # Dynamic BASE_DIR (folder tempat file script ini berada: /preprocessing)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load Config dengan path yang aman
    config_path = os.path.join(BASE_DIR, "config.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Helper function untuk memastikan path selalu tepat dari BASE_DIR
    def get_path(path_str):
        if os.path.isabs(path_str):
            return path_str
        return os.path.normpath(os.path.join(BASE_DIR, path_str))

    # Load Data Raw
    raw_path = get_path(config["paths"]["raw_data_path"])
    df = pd.read_csv(raw_path)

    # Drop Identifier & Split Features/Target
    df_cleaned = df.drop(columns=config["features"]["drop_columns"], errors="ignore")
    X = df_cleaned.drop(columns=[config["features"]["target_column"]])
    y = df_cleaned[config["features"]["target_column"]]

    # Train-Test Split (Stratified) + .copy() untuk hindari SettingWithCopyWarning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["params"]["test_size"],
        random_state=config["params"]["random_state"],
        stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Log Transformation
    for col in config["features"]["skewed_columns"]:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])

    # One-Hot Encoding
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="error")
    cat_cols = config["features"]["categorical_columns"]
    
    encoded_train = ohe.fit_transform(X_train[cat_cols])
    encoded_test = ohe.transform(X_test[cat_cols])
    
    encoded_cols = ohe.get_feature_names_out(cat_cols)
    df_enc_train = pd.DataFrame(encoded_train, columns=encoded_cols, index=X_train.index)
    df_enc_test = pd.DataFrame(encoded_test, columns=encoded_cols, index=X_test.index)
    
    X_train = pd.concat([X_train.drop(columns=cat_cols), df_enc_train], axis=1)
    X_test = pd.concat([X_test.drop(columns=cat_cols), df_enc_test], axis=1)

    # Scaling
    scaler = StandardScaler()
    num_cols = config["features"]["scaling_columns"]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Save Processed Datasets & Artifacts
    out_dir = get_path(config["paths"]["output_dir"])
    art_dir = get_path(config["paths"]["artifact_dir"])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)
    
    joblib.dump(scaler, os.path.join(art_dir, "scaler.pkl"))
    joblib.dump(ohe, os.path.join(art_dir, "encoder.pkl"))
    print("[SUCCESS] Preprocessing berhasil dijalankan dan data siap dilatih.")

if __name__ == "__main__":
    main()