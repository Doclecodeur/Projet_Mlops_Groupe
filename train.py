# train.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ── Features ─────────────────────────────────────────
FEATURES = [
    'credit_lines_outstanding',
    'loan_amt_outstanding',
    'total_debt_outstanding',
    'income',
    'years_employed',
    'fico_score'
]
TARGET = 'default'


def load_data():
    df = pd.read_csv("Loan_Data.csv")
    df.dropna(inplace=True)
    df.drop(columns=['customer_id'], inplace=True)
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def train():
    X, y = load_data()

    print(f" Dataset : {X.shape[0]} lignes")
    print(f" Distribution :\n{y.value_counts()}")

    # ── Split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── SMOTE ─────────────────────────────────────────
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f" Après SMOTE : {X_train_res.shape[0]} échantillons")

    # ── Scaler ────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc = scaler.transform(X_test)

    # ── Modèles ───────────────────────────────────────
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
    }

    best_auc = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_train_sc, y_train_res)
        y_pred = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"\n {name}")
        print(f"   Accuracy : {acc:.4f}")
        print(f"   F1 Score : {f1:.4f}")
        print(f"   ROC AUC  : {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    # ── Sauvegarde ────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(FEATURES, "models/features.pkl")

    print(f"\nMeilleur modèle : {best_name} (AUC={best_auc:.4f})")
    print(" Sauvegardé dans models/")


if __name__ == "__main__":
    train()
