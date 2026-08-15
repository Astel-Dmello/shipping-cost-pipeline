from config import DataConfig, ValidationConfig
from data_generator import generate
from evaluate import evaluate_model
from feature_selection import correlation_selection
from preprocessing import build_variants


def test_evaluate_model_returns_valid_r2():
    raw = generate(DataConfig(n_rows=800, seed=4))
    variants = build_variants(raw)
    df = variants["listwise_deletion"]
    cols, _ = correlation_selection(df)

    small_config = ValidationConfig(n_folds=3, dnn_epochs=3)
    result = evaluate_model("MLR", df, cols, "test_branch", small_config)

    assert -1.0 <= result["r_squared"] <= 1.0
    assert result["training_time_s"] >= 0
    assert len(result["sample_predictions"]) == 5
