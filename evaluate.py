"""
Validation protocol matching the paper (Sec 5.1, Figure 2):
  - 80/20 train/test split.
  - Training set further split into K folds; each fold-model is trained and used to
    predict the held-out 20% test set, producing K predictions per test point.
  - A 95% confidence interval over those K predictions gives the predicted cost range;
    their mean gives the point prediction used for the R^2 comparison table.

The paper uses K=30 folds and 500-epoch DNN training on ~1M rows. Reproducing that
exactly on a laptop-scale synthetic demo would take a very long time for no extra
fidelity value, so config.py's demo defaults use fewer folds/epochs. Pass a
config.get_config(faithful=True) to run the paper's actual protocol values instead.
"""
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, train_test_split

from config import ValidationConfig
from feature_selection import numeric_frame, TARGET
from logging_config import get_logger
from models import MODEL_REGISTRY

logger = get_logger(__name__)


def _prep_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    work = numeric_frame(df)[feature_cols + [TARGET]].dropna()
    X = work[feature_cols].values.astype(float)
    y = work[TARGET].values.astype(float)
    return X, y


def _fit_predict(
    model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
    config: ValidationConfig, input_dim: int,
) -> np.ndarray:
    spec = MODEL_REGISTRY[model_name]
    if spec["keras"]:
        model = spec["builder"](input_dim=input_dim)
        model.fit(X_train, y_train, epochs=config.dnn_epochs, batch_size=256, verbose=0)
        return model.predict(X_val, verbose=0).ravel()
    model = spec["builder"](small_sample_safe=config.small_sample_safe_boosting)
    model.fit(X_train, y_train)
    return model.predict(X_val)


def evaluate_model(
    model_name: str, df: pd.DataFrame, feature_cols: List[str], branch_label: str,
    config: ValidationConfig = ValidationConfig(),
) -> Dict:
    """
    Runs the paper's protocol for one model on one (selection_method, preprocessing)
    branch: 80/20 split, K-fold training on the 80%, K predictions per test row,
    R^2 on the mean prediction, and a 95% CI per test row.
    """
    X, y = _prep_xy(df, feature_cols)
    if len(X) < config.n_folds * 2:
        raise ValueError(
            f"Not enough rows ({len(X)}) for {config.n_folds}-fold CV on branch {branch_label}"
        )

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    fold_preds = np.zeros((config.n_folds, len(X_test)))

    start = time.time()
    for i, (train_idx, _val_idx) in enumerate(kf.split(X_train_full)):
        fold_preds[i] = _fit_predict(
            model_name, X_train_full[train_idx], y_train_full[train_idx], X_test,
            config, input_dim=X.shape[1],
        )
    elapsed = time.time() - start

    mean_pred = fold_preds.mean(axis=0)
    ss_res = np.sum((y_test - mean_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    sem = fold_preds.std(axis=0, ddof=1) / np.sqrt(config.n_folds)
    t_crit = stats.t.ppf((1 + config.confidence) / 2, df=config.n_folds - 1)
    ci_lower = mean_pred - t_crit * sem
    ci_upper = mean_pred + t_crit * sem

    return {
        "model": model_name,
        "branch": branch_label,
        "n_features": len(feature_cols),
        "r_squared": round(float(r2), 3),
        "training_time_s": round(elapsed, 2),
        "sample_predictions": pd.DataFrame({
            "actual": y_test[:5],
            "predicted": mean_pred[:5].round(0),
            "ci_lower": ci_lower[:5].round(0),
            "ci_upper": ci_upper[:5].round(0),
        }),
    }


def run_comparison(
    variants: Dict[str, pd.DataFrame], branches: Dict[Tuple[str, str], Dict],
    config: ValidationConfig = ValidationConfig(),
) -> pd.DataFrame:
    """
    variants: {preprocessing_variant_name: preprocessed_df}
    branches: output of feature_selection.build_feature_branches -- keyed by
              (selection_method, preprocessing_variant), each holding "features"
              (used by XGBoost/LightGBM/DNN) and "mlr_features" (VIF-reduced, MLR only).
    Returns a comparison table shaped like the paper's Table 14.
    """
    rows = []
    for (method_name, variant_name), branch in branches.items():
        df = variants[variant_name]
        branch_label = f"{method_name}+{variant_name}"
        for model_name in MODEL_REGISTRY:
            cols = branch["mlr_features"] if model_name == "MLR" else branch["features"]
            try:
                result = evaluate_model(model_name, df, cols, branch_label, config)
            except ValueError as e:
                logger.warning("Skipping %s / %s: %s", model_name, branch_label, e)
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
                model_name, branch_label, result["n_features"],
                result["r_squared"], result["training_time_s"],
            )
    return pd.DataFrame(rows)
