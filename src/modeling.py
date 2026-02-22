"""
Modeling — train / evaluate six regression models for hedonic price prediction.

Models: Linear Regression, Ridge, Lasso, Random Forest, XGBoost, CatBoost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

from .config import (
    RANDOM_STATE,
    TEST_SIZE,
    KFOLD_SPLITS,
    RF_BEST_PARAMS,
    XGB_BEST_PARAMS,
    CATBOOST_BEST_PARAMS,
)


# ==========================================================================
# Data Preparation
# ==========================================================================
def prepare_features(
    df: pd.DataFrame,
    target_col: str = "log_price",
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Separate features / target and scale features.

    Returns (X_scaled, y, scaler).
    """
    drop_cols = drop_cols or ["id", "log_price", "estimated_revenue"]
    drop_existing = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_existing)
    y = df[target_col]

    # Keep only numeric columns and fill any remaining NaN
    X = X.select_dtypes(include="number").fillna(0)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_scaled, y, scaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """80/20 train-test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


# ==========================================================================
# Individual Model Trainers
# ==========================================================================
def train_linear_regression(X_train, y_train) -> LinearRegression:
    """Fit an OLS linear regression (baseline)."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge(
    X_train,
    y_train,
    alphas: np.ndarray | None = None,
) -> tuple[Ridge, pd.DataFrame]:
    """Fit Ridge regression over a range of alpha values.

    Returns the best-fit model and a coefficient trace DataFrame.
    """
    alphas = alphas if alphas is not None else np.arange(0, 2000, 1)

    best_r2, best_model = -np.inf, None
    coef_records = []

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        r2 = ridge.score(X_train, y_train)
        coef_records.append(
            {"alpha": alpha, "r2": r2, **dict(zip(X_train.columns, ridge.coef_))}
        )
        if r2 > best_r2:
            best_r2, best_model = r2, ridge

    trace_df = pd.DataFrame(coef_records)
    return best_model, trace_df


def train_lasso(
    X_train,
    y_train,
    alphas: np.ndarray | None = None,
) -> tuple[Lasso, pd.DataFrame]:
    """Fit Lasso regression over a range of alpha values.

    Returns the best-fit model and a coefficient trace DataFrame.
    """
    alphas = alphas if alphas is not None else np.arange(0.01, 8.02, 0.02)

    best_r2, best_model = -np.inf, None
    coef_records = []

    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10_000)
        lasso.fit(X_train, y_train)
        r2 = lasso.score(X_train, y_train)
        coef_records.append(
            {"alpha": alpha, "r2": r2, **dict(zip(X_train.columns, lasso.coef_))}
        )
        if r2 > best_r2:
            best_r2, best_model = r2, lasso

    trace_df = pd.DataFrame(coef_records)
    return best_model, trace_df


def train_random_forest(
    X_train,
    y_train,
    params: dict | None = None,
) -> RandomForestRegressor:
    """Train a Random Forest with pre-tuned hyperparameters."""
    params = params or RF_BEST_PARAMS
    model = RandomForestRegressor(**params, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, params: dict | None = None):
    """Train an XGBoost regressor with pre-tuned hyperparameters."""
    import xgboost as xgb

    params = params or XGB_BEST_PARAMS
    model = xgb.XGBRegressor(**params, random_state=RANDOM_STATE)
    model.fit(X_train, y_train, verbose=False)
    return model


def train_catboost(X_train, y_train, params: dict | None = None):
    """Train a CatBoost regressor with pre-tuned hyperparameters."""
    from catboost import CatBoostRegressor

    params = params or CATBOOST_BEST_PARAMS
    model = CatBoostRegressor(**params, random_state=RANDOM_STATE, verbose=0)
    model.fit(X_train, y_train)
    return model


# ==========================================================================
# Evaluation Helpers
# ==========================================================================
def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
) -> dict:
    """Compute R², MSE, and MAE for a fitted model on train & test sets."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return {
        "train_r2":  round(r2_score(y_train, train_pred), 4),
        "test_r2":   round(r2_score(y_test, test_pred), 4),
        "train_mse": round(mean_squared_error(y_train, train_pred), 4),
        "test_mse":  round(mean_squared_error(y_test, test_pred), 4),
        "train_mae": round(mean_absolute_error(y_train, train_pred), 4),
        "test_mae":  round(mean_absolute_error(y_test, test_pred), 4),
        "train_pred": train_pred,
        "test_pred":  test_pred,
    }


def run_kfold_cv(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    k: int = KFOLD_SPLITS,
) -> pd.DataFrame:
    """Run K-Fold cross-validation for multiple models.

    Parameters
    ----------
    models : dict
        ``{name: fitted_model}`` mapping.
    X, y : array-like
        Full (unscaled) feature matrix and target vector.
    k : int
        Number of folds.

    Returns
    -------
    pd.DataFrame with columns: model, mean_r2, std_r2, mean_mse, std_mse.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["r2", "neg_mean_squared_error"]
    records = []

    for name, model in models.items():
        cv = cross_validate(model, X, y, cv=kf, scoring=scoring)
        records.append({
            "model":    name,
            "mean_r2":  round(cv["test_r2"].mean(), 4),
            "std_r2":   round(cv["test_r2"].std(), 4),
            "mean_mse": round(-cv["test_neg_mean_squared_error"].mean(), 4),
            "std_mse":  round(cv["test_neg_mean_squared_error"].std(), 4),
        })

    return pd.DataFrame(records)


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Create a comparison DataFrame from ``{name: evaluate_model() result}``."""
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model":     name,
            "Train R²":  metrics["train_r2"],
            "Test R²":   metrics["test_r2"],
            "Train MSE": metrics["train_mse"],
            "Test MSE":  metrics["test_mse"],
            "Train MAE": metrics["train_mae"],
            "Test MAE":  metrics["test_mae"],
        })
    return pd.DataFrame(rows).set_index("Model")


# ==========================================================================
# Feature Importance
# ==========================================================================
def get_feature_importance(model, feature_names) -> pd.DataFrame:
    """Extract feature importances from tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model does not expose feature importances.")

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
