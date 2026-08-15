"""
Full pipeline entry point. Reproduces the STRUCTURE of Jang, Chang & Kim (2023)'s
freight shipping-cost prediction study on synthetic data:

  generate synthetic data -> preprocess (2 variants) -> select features (correlation
  filter + stepwise, 4 branches total) -> train & evaluate 4 models per branch with a
  K-fold confidence interval -> comparison table (cf. paper Table 14) -> SHAP feature
  importance -> save results/plots to results/.

Usage:
  python main.py                 # demo settings (fast, ~1-2 min)
  python main.py --faithful      # paper's actual protocol (30-fold CV, 500-epoch DNN;
                                  # much slower, run with a coffee break)
  python main.py --n-rows 20000  # override synthetic dataset size
"""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config
from data_generator import generate
from evaluate import run_comparison
from feature_importance import shap_importance
from feature_selection import build_feature_branches
from logging_config import get_logger
from preprocessing import build_variants

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faithful", action="store_true",
                         help="Use the paper's actual protocol (30-fold CV, 500 DNN epochs). Slow.")
    parser.add_argument("--n-rows", type=int, default=None,
                         help="Override number of synthetic rows generated (default: 8000).")
    parser.add_argument("--real-data", type=str, default=None,
                         help="Path to a real dataset CSV (e.g. data/SCMS_Delivery_History_"
                              "Dataset.csv) to run the pipeline on instead of synthetic data. "
                              "See real_data.py for the SCMS-dataset-specific cleaning this "
                              "applies; a different real CSV would need its own loader.")
    return parser.parse_args()


def save_plots(results, importance, share, results_dir: Path):
    results_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = results.pivot(index="model", columns="branch", values="r_squared")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("R-squared")
    ax.set_title("Model comparison across preprocessing/selection branches")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    share.sort_values().plot(kind="barh", ax=ax, color="#2f6fed")
    ax.set_xlabel("Share of total SHAP contribution (%)")
    ax.set_title("Feature importance (LightGBM, SHAP) - cf. paper Figure 3")
    fig.tight_layout()
    fig.savefig(results_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    logger.info("Saved plots to %s/", results_dir)


def main():
    args = parse_args()
    config = get_config(faithful=args.faithful)
    if args.n_rows:
        config = replace(config, data=replace(config.data, n_rows=args.n_rows))

    logger.info("=" * 70)
    logger.info("STEP 1/5: Loading data")
    logger.info("=" * 70)
    if args.real_data:
        from real_data import load_real_data, real_continuous_columns
        raw = load_real_data(args.real_data)
        continuous_cols = real_continuous_columns()
        logger.info("Using REAL data: %s (%d rows)", args.real_data, len(raw))
    else:
        raw = generate(config.data)
        continuous_cols = None  # preprocessing.py's default (synthetic schema)
        logger.info("Using SYNTHETIC data (%d rows)", len(raw))

    logger.info("=" * 70)
    logger.info("STEP 2/5: Preprocessing (outlier removal + 2 missing-data variants)")
    logger.info("=" * 70)
    variants = build_variants(raw, config.preprocess, continuous_columns=continuous_cols)

    logger.info("=" * 70)
    logger.info("STEP 3/5: Feature selection (correlation + stepwise, x 2 preprocessing variants)")
    logger.info("=" * 70)
    branches = build_feature_branches(variants, config.feature_selection)

    logger.info("=" * 70)
    logger.info("STEP 4/5: Training & evaluating models across all branches")
    logger.info("=" * 70)
    mode = "small-sample-safe (deviates from paper, see models.py)" \
        if config.validation.small_sample_safe_boosting else "paper's literal hyperparameters"
    logger.info("Boosting hyperparameter mode: %s", mode)
    results = run_comparison(variants, branches, config.validation)

    logger.info("=" * 70)
    logger.info("COMPARISON TABLE (cf. paper Table 14)")
    logger.info("=" * 70)
    pivot = results.pivot(index="model", columns="branch", values="r_squared")
    print(pivot.to_string())

    best_row = results.loc[results["r_squared"].idxmax()]
    logger.info("Best: %s on %s (R^2=%.3f)",
                best_row["model"], best_row["branch"], best_row["r_squared"])

    boosting_best = results[results["model"].isin(["XGBoost", "LightGBM"])]["r_squared"].max()
    mlr_best = results[results["model"] == "MLR"]["r_squared"].max()
    if boosting_best > mlr_best:
        logger.info("Boosting models outperform MLR (best boosting R^2=%.3f > best MLR R^2=%.3f), "
                     "matching the paper's finding.", boosting_best, mlr_best)
    else:
        logger.warning("MLR (R^2=%.3f) still matches/beats boosting (R^2=%.3f) on this synthetic "
                        "run -- see README's 'Comparing to the paper' section.", mlr_best, boosting_best)

    logger.info("=" * 70)
    logger.info("STEP 5/5: Feature importance (SHAP, cf. paper Figure 3 / Sec 5.7)")
    logger.info("=" * 70)
    step_branch = branches.get(("stepwise", "listwise_deletion")) \
        or next(iter(branches.values()))
    importance, share = shap_importance(variants["listwise_deletion"], step_branch["features"])
    for feat in importance.index:
        logger.info("  %-20s %5.1f%%", feat, share[feat])

    RESULTS_DIR = Path("results_real") if args.real_data else Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    save_plots(results, importance, share, RESULTS_DIR)

    data_desc = f"REAL data ({args.real_data})" if args.real_data else "SYNTHETIC data (see 03_synthetic_data_design.yaml)"
    logger.info("Done. Results saved to %s/. Ran on %s -- expect the same qualitative "
                "pattern as the paper's findings, not necessarily identical R^2 values.",
                RESULTS_DIR, data_desc)


if __name__ == "__main__":
    main()
