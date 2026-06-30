"""
Shared modeling utilities: the preprocessing ColumnTransformer and the model
zoo. Centralizing these keeps training, evaluation, explainability, and the
prediction interface perfectly consistent.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.feature_engineering import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


def build_preprocessor() -> ColumnTransformer:
    """Numeric -> impute+scale, categorical -> impute+one-hot, binary -> pass-through."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )


def build_models() -> dict:
    """The model zoo required by the spec. KNN is the baseline."""
    rs = config.RANDOM_STATE
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=rs),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=rs),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1,
            class_weight="balanced", random_state=rs),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
        "KNN (baseline)": KNeighborsClassifier(n_neighbors=15),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=rs, n_jobs=-1, tree_method="hist",
        )
    return models


def make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor()),
        ("clf", model),
    ])
