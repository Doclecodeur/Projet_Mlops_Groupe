import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os

# ── Configuration MLflow ─────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("credit-risk-experiment")

# ── Features (sans customer_id) ──────────────────────
FEATURES = [
    'credit_lines_outstanding',
    'loan_amt_outstanding',
    'total_debt_outstanding',
    'income',
    'years_employed',
    'fico_score'
]
TARGET = 'default'


def load_and_prepare_data():
    df = pd.read_csv("data/Loan_Data.csv")
    df.dropna(inplace=True)
    df.drop(columns=['customer_id'], inplace=True)  # inutile pour le modèle
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def train_with_mlflow():
    X, y = load_and_prepare_data()

    print(f" Dataset chargé : {X.shape[0]} lignes, {X.shape[1]} features")
    print(f" Distribution target :\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── SMOTE ────────────────────────────────────────
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f" Après SMOTE : {X_train_res.shape[0]} échantillons")

    # ── Scaler ───────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc = scaler.transform(X_test)

    # ── Modèles ──────────────────────────────────────
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
    }

    best_auc = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            # ── Params ───────────────────────────────
            mlflow.log_param("model", name)
            mlflow.log_param("smote", True)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("features", FEATURES)

            # ── Entraînement ─────────────────────────
            model.fit(X_train_sc, y_train_res)
            y_pred = model.predict(X_test_sc)
            y_proba = model.predict_proba(X_test_sc)[:, 1]

            # ── Métriques ────────────────────────────
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)

            print(f"\n {name}")
            print(f"   Accuracy : {acc:.4f}")
            print(f"   F1 Score : {f1:.4f}")
            print(f"   ROC AUC  : {auc:.4f}")

            # ── Log modèle ───────────────────────────
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            # ── Meilleur modèle ──────────────────────
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_name = name

    # ── Sauvegarde du meilleur modèle ────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(FEATURES, "models/features.pkl")

    print(f"\n Meilleur modèle : {best_name} (AUC={best_auc:.4f})")
    print(" Sauvegardé dans models/best_model.pkl")


if __name__ == "__main__":
    train_with_mlflow()
