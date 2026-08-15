"""
Model builders, one per model the paper compares (Sec 5.2-5.5), with the paper's
stated hyperparameters. Where the paper is silent, a comment documents the default used.
"""
from typing import Any

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def build_mlr() -> LinearRegression:
    """Multiple linear regression -- no special hyperparameters stated (paper Sec 5.2)."""
    return LinearRegression()


def build_xgboost(small_sample_safe: bool = False) -> XGBRegressor:
    """XGBoost, 'basic form': 400 weak learners, max depth 3, learning rate 0.3
    (paper Sec 5.4).

    Note: those hyperparameters were tuned by the paper's authors against ~1M rows.
    Applied literally to a much smaller synthetic sample (the default here), 400 trees
    at lr=0.3 readily overfits -- R^2 actually falls as more features are added, the
    classic overfitting signature. Pass small_sample_safe=True for a documented,
    intentional deviation (fewer/shallower trees, lower learning rate) suited to
    small-n demo runs; this is NOT what the paper used, and main.py logs which mode
    is active so results aren't silently non-comparable."""
    if small_sample_safe:
        return XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", verbosity=0,
        )
    return XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.3,
        objective="reg:squarederror",
        verbosity=0,
    )


def build_lightgbm(small_sample_safe: bool = False) -> LGBMRegressor:
    """LightGBM, 'basic form': learning rate 0.1, rest default (paper Sec 5.5).
    See build_xgboost's docstring for what small_sample_safe changes and why."""
    if small_sample_safe:
        return LGBMRegressor(
            n_estimators=150, learning_rate=0.05, num_leaves=15,
            min_child_samples=20, verbosity=-1,
        )
    return LGBMRegressor(learning_rate=0.1, verbosity=-1)


def build_dnn(input_dim: int) -> Any:
    """5 hidden layers (256->128->64->32->16), Adam, lr 0.001 (paper Table 7).
    Returns a compiled Keras model. Epoch/batch settings are applied at fit time
    (see evaluate.py / config.py) rather than baked in here."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


MODEL_REGISTRY = {
    "MLR": {"builder": lambda **_: build_mlr(), "keras": False},
    "XGBoost": {"builder": lambda small_sample_safe=False, **_: build_xgboost(small_sample_safe), "keras": False},
    "LightGBM": {"builder": lambda small_sample_safe=False, **_: build_lightgbm(small_sample_safe), "keras": False},
    "DNN": {"builder": lambda input_dim, **_: build_dnn(input_dim), "keras": True},
}
