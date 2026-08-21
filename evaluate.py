"""
Model evaluation for the shipping-cost prediction pipeline.

Improvements for real SCMS data:
  - 80/20 train-test split.
  - K-fold ensemble training on the training portion.
  - log1p transformation of the highly skewed shipping_cost target.
  - Predictions converted back to original USD scale before R² calculation.
  - StandardScaler used for DNN features.
  - Mean prediction across fold models used as final prediction.
  - 95% confidence intervals calculated from fold predictions.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from config import ValidationConfig
from feature_selection import numeric_frame, TARGET
from logging_config import get_logger
from models import MODEL_REGISTRY

logger = get_logger(__name__)


def _prep_xy(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:

    numeric_df = numeric_frame(df)

    required_cols = [c for c in feature_cols if c in numeric_df.columns]

    if TARGET not in numeric_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")

    work = numeric_df[required_cols + [TARGET]].copy()

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna()

    X = work[required_cols].astype(float).values
    y = work[TARGET].astype(float).values

    # Safety: shipping cost must be non-negative
    valid = y >= 0
    X = X[valid]
    y = y[valid]

    return X, y


def _fit_predict(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: ValidationConfig,
    input_dim: int,
) -> np.ndarray:

    spec = MODEL_REGISTRY[model_name]

    # Shipping cost has a very large right-skew.
    # Train models on log scale.
    y_train_log = np.log1p(y_train)

    if spec["keras"]:

        # DNN performs much better when numerical inputs are standardized.
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = spec["builder"](input_dim=input_dim)

        model.fit(
            X_train_scaled,
            y_train_log,
            epochs=config.dnn_epochs,
            batch_size=128,
            verbose=0,
        )

        log_predictions = model.predict(
            X_test_scaled,
            verbose=0
        ).ravel()

    else:

        model = spec["builder"](
            small_sample_safe=config.small_sample_safe_boosting
        )

        model.fit(X_train, y_train_log)

        log_predictions = model.predict(X_test)

    # Convert predictions from log scale back to USD scale
    predictions = np.expm1(log_predictions)

    # Shipping cost cannot be negative
    predictions = np.maximum(predictions, 0)

    return predictions


def evaluate_model(
    model_name: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    branch_label: str,
    config: ValidationConfig = ValidationConfig(),
) -> Dict:

    """
    Evaluates one model using:
      1. 80/20 train-test split
      2. K-fold ensemble on training data
      3. Mean prediction from all folds
      4. R² calculated on original shipping-cost scale
    """

    X, y = _prep_xy(df, feature_cols)

    if len(X) < config.n_folds * 2:
        raise ValueError(
            f"Not enough rows ({len(X)}) for "
            f"{config.n_folds}-fold CV on branch {branch_label}"
        )

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    kf = KFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    fold_preds = np.zeros(
        (config.n_folds, len(X_test))
    )

    start = time.time()

    for i, (train_idx, val_idx) in enumerate(
        kf.split(X_train_full)
    ):

        X_fold_train = X_train_full[train_idx]
        y_fold_train = y_train_full[train_idx]

        fold_preds[i] = _fit_predict(
            model_name=model_name,
            X_train=X_fold_train,
            y_train=y_fold_train,
            X_test=X_test,
            config=config,
            input_dim=X.shape[1],
        )

    elapsed = time.time() - start

    # Final prediction = average prediction from all fold models
    mean_pred = fold_preds.mean(axis=0)

    # R² on ORIGINAL shipping cost scale
    ss_res = np.sum(
        (y_test - mean_pred) ** 2
    )

    ss_tot = np.sum(
        (y_test - y_test.mean()) ** 2
    )

    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)

    # Confidence interval
    if config.n_folds > 1:

        sem = (
            fold_preds.std(axis=0, ddof=1)
            / np.sqrt(config.n_folds)
        )

        t_crit = stats.t.ppf(
            (1 + config.confidence) / 2,
            df=config.n_folds - 1,
        )

        ci_lower = mean_pred - (t_crit * sem)
        ci_upper = mean_pred + (t_crit * sem)

        ci_lower = np.maximum(ci_lower, 0)

    else:
        ci_lower = mean_pred.copy()
        ci_upper = mean_pred.copy()

    return {
        "model": model_name,
        "branch": branch_label,
        "n_features": len(feature_cols),
        "r_squared": round(float(r2), 3),
        "training_time_s": round(elapsed, 2),

        "sample_predictions": pd.DataFrame({
            "actual": y_test[:5].round(2),
            "predicted": mean_pred[:5].round(2),
            "ci_lower": ci_lower[:5].round(2),
            "ci_upper": ci_upper[:5].round(2),
        }),
    }


def run_comparison(
    variants: Dict[str, pd.DataFrame],
    branches: Dict[Tuple[str, str], Dict],
    config: ValidationConfig = ValidationConfig(),
) -> pd.DataFrame:

    """
    Runs all models across all preprocessing and feature-selection branches.
    """

    rows = []

    for (method_name, variant_name), branch in branches.items():

        df = variants[variant_name]

        branch_label = (
            f"{method_name}+{variant_name}"
        )

        for model_name in MODEL_REGISTRY:

            # MLR uses VIF-reduced features.
            # Other models use the complete selected feature set.
            if model_name == "MLR":
                cols = branch["mlr_features"]
            else:
                cols = branch["features"]

            try:

                result = evaluate_model(
                    model_name=model_name,
                    df=df,
                    feature_cols=cols,
                    branch_label=branch_label,
                    config=config,
                )

            except ValueError as e:

                logger.warning(
                    "Skipping %s / %s: %s",
                    model_name,
                    branch_label,
                    e,
                )

                continue

            rows.append({
                "model": result["model"],
                "branch": result["branch"],
                "n_features": result["n_features"],
                "r_squared": result["r_squared"],
                "training_time_s": result["training_time_s"],
            })

            logger.info(
                "%-10s | %-28s | %2d feats | R2=%.3f | %.1fs",
                model_name,
                branch_label,
                result["n_features"],
                result["r_squared"],
                result["training_time_s"],
            )

    return pd.DataFrame(rows)