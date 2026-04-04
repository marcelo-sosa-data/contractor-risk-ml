"""
train_model.py
Entrenamiento del modelo de riesgo de contratistas con XGBoost.
Ejecutar: python src/train_model.py
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Features ──────────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "penalty_rate", "avg_penalty_amount", "penalty_trend",
    "late_delivery_rate", "avg_delay_days", "log_contract_count",
    "years_as_contractor", "total_value_log", "risk_index",
]
CATEGORICAL_FEATURES = ["contractor_type", "region", "specialization"]
BINARY_FEATURES      = ["single_supplier"]
TARGET               = "has_penalty"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering desde datos raw de contratistas."""
    df = df.copy()
    df["penalty_rate"]       = df["total_penalties"] / df["total_contracts"].clip(lower=1)
    df["avg_penalty_amount"] = df["total_penalty_amount"] / df["total_penalties"].clip(lower=1)
    df["penalty_trend"]      = df["total_penalties"] / df["years_as_contractor"].clip(lower=0.1)
    df["late_delivery_rate"] = df["late_deliveries"] / df["total_contracts"].clip(lower=1)
    df["avg_delay_days"]     = df["total_delay_days"] / df["late_deliveries"].clip(lower=1)
    df["log_contract_count"] = np.log1p(df["total_contracts"])
    df["total_value_log"]    = np.log1p(df["total_contract_value"])
    df["risk_index"]         = (
        df["penalty_rate"] * 0.4 +
        df["late_delivery_rate"] * 0.3 +
        (df["penalty_trend"] > df["penalty_trend"].median()).astype(float) * 0.3
    )
    return df


def build_pipeline() -> Pipeline:
    """Pipeline sklearn con preprocessing + XGBoost."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,   NUMERIC_FEATURES + BINARY_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def evaluate(pipeline, X_test, y_test, threshold=0.35):
    """Evaluación con umbral ajustado al negocio."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n{'═'*50}")
    print(f"  ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  PR-AUC  : {average_precision_score(y_test, y_prob):.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Bajo Riesgo','Alto Riesgo'])}")
    print(f"{'═'*50}")


if __name__ == "__main__":
    # Cargar datos
    df = pd.read_csv("data/sample/contractors_sample.csv")
    df = build_features(df)

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    X = df[[f for f in all_features if f in df.columns]]
    y = df[TARGET]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cross-validation
    pipe = build_pipeline()
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_r = cross_validate(pipe, X_train, y_train,
                          cv=cv, scoring=["roc_auc","average_precision"], n_jobs=-1)
    print(f"CV ROC-AUC: {cv_r['test_roc_auc'].mean():.4f} ± {cv_r['test_roc_auc'].std():.4f}")

    # Entrenar final
    pipe.fit(X_train, y_train)
    evaluate(pipe, X_test, y_test)

    # Guardar modelo
    joblib.dump(pipe, "models/xgb_risk_model_v1.pkl")
    print("✅ Modelo guardado en models/xgb_risk_model_v1.pkl")
