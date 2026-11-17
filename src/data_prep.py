import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

def prepare_data(csv_path):
    print("[1/4] Loading dataset...")
    df = pd.read_csv(csv_path)

    # Non-predictive IDs aur leakage columns hatana
    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns]).dropna()

    # Target column identify karna
    target_col = 'isFraud' if 'isFraud' in df.columns else 'Class'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Categorical encoding (e.g., 'type' column)
    print("[2/4] Encoding categorical features...")
    X = pd.get_dummies(X, drop_first=True)
    joblib.dump(list(X.columns), "models/feature_columns.pkl")

    # Train-Test Split (Data leakage rokne ke liye scaling se pehle)
    print("[3/4] Splitting & scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "models/robust_scaler.pkl")

    # 2 Bank Nodes (Bank 1 & Bank 2) me split + Local SMOTE
    print("[4/4] Partitioning into Bank Nodes & applying SMOTE...")
    splits = np.array_split(np.arange(len(X_train_scaled)), 2)
    for idx, indices in enumerate(splits):
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train_scaled[indices], y_train.iloc[indices])
        np.savez(f"data/processed/bank_{idx+1}_data.npz", X=X_res, y=y_res)
        print(f"[OK] Bank {idx+1} Ready: {X_res.shape[0]} samples")

    np.savez("data/processed/global_test.npz", X=X_test_scaled, y=y_test.values)
    print("[SUCCESS] Preprocessing Complete!")

if __name__ == "__main__":
    prepare_data("data/raw/PS_20174392719_1491204439457_log.csv")
