"""
Modeling script for the heart disease dataset.

Builds a Logistic Regression classifier with preprocessing
and prints evaluation metrics.

Run:
    python -m src.modeling
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from .data_utils import load_heart_data


def build_model():
    # For reference we assume same schema as in EDA:
    df = load_heart_data()

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    numeric_features = X.select_dtypes(exclude="object").columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    return clf, X, y


def main():
    clf, X, y = build_model()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.3f}")

    # Show top positive/negative coefficients (feature importance for LR)
    model = clf.named_steps["model"]
    pre = clf.named_steps["preprocess"]

    # Get feature names
    num_cols = pre.transformers_[0][2]
    cat_transformer = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_cols = cat_transformer.get_feature_names_out(pre.transformers_[1][2])
    feature_names = list(num_cols) + list(cat_cols)

    coefs = model.coef_[0]
    coef_series = pd.Series(coefs, index=feature_names).sort_values(ascending=False)

    print("\n=== Top 10 risk-increasing features (positive coefficients) ===")
    print(coef_series.head(10))

    print("\n=== Top 10 risk-decreasing features (negative coefficients) ===")
    print(coef_series.tail(10))


if __name__ == "__main__":
    main()
