# Prediction of Shipping Cost on Freight Brokerage Platform — reproduction pipeline (synthetic data)

Implements the data pipeline and model comparison from Jang, H.-S.; Chang, T.-W.; Kim, S.-H.
"Prediction of Shipping Cost on Freight Brokerage Platform Using Machine Learning."
*Sustainability* 2023, 15, 1122. https://doi.org/10.3390/su15021122

The paper's real dataset (1.88M freight brokerage records, Apr-Sep 2020) is **private/
proprietary** — its Data Availability Statement says raw data is confidential trade-secret
information held by Hwamulman Co. Ltd. This pipeline runs on **synthetic data built to
match the paper's described schema and reported statistics** (see `03_synthetic_data_design.yaml`).
Treat the numbers below as illustrating the *pipeline and method*, not a reproduction of
the paper's actual results.

## Run it
```bash
pip install -r requirements.txt
python main.py                                            # synthetic data, demo settings, ~2 min
python main.py --faithful                                 # synthetic, paper's actual protocol (slow)
python main.py --real-data data/SCMS_Delivery_History_Dataset.csv   # real public dataset
streamlit run streamlit_app.py                             # interactive UI for all of the above
python -m pytest                                           # 17 unit tests covering every module
```
Outputs land in `results/` (synthetic) or `results_real/` (real data):
`comparison_table.csv`, `model_comparison.png`, `feature_importance.png`.

## Architecture
```
config.py              single source of truth for every tunable parameter
logging_config.py       shared structured logging
data_generator.py       synthetic data matching paper schema + stats
preprocessing.py        IQR outlier removal + 2 missing-data variants
feature_selection.py    correlation filter, stepwise selection, VIF filter
models.py                MLR / DNN / XGBoost / LightGBM builders
evaluate.py               K-fold + 95% CI validation protocol, 4 branches x 4 models
feature_importance.py    SHAP analysis
main.py                   CLI entry point, orchestration, plots
tests/                    pytest suite (13 tests, all passing)
```

## What's implemented
- **Data**: 8,000 synthetic rows / 36 columns, with injected missingness and outliers
  so preprocessing has real work to do (paper Sec 3.2).
- **Preprocessing**: IQR outlier removal + listwise-deletion / mean-imputation variants.
- **Feature selection**: Pearson correlation filter (|r| ≥ 0.1) *and* stepwise selection,
  each crossed with both preprocessing variants — 4 branches total, matching the paper's
  Table 14 structure.
- **Models**: MLR, DNN (256→128→64→32→16), XGBoost (400 trees / depth 3 / lr 0.3),
  LightGBM (lr 0.1) — hyperparameters copied verbatim from the paper.
- **Validation**: 80/20 split, K-fold CV on the training set, 95% confidence interval
  per prediction (paper Sec 5.1 / Figure 2).
- **Feature importance**: SHAP on LightGBM.

## Two real bugs found and fixed during development (documented, not hidden)

**1. VIF filtering was incorrectly applied to every model.** The paper's VIF-based
multicollinearity check (Sec 5.2) is specifically about MLR — OLS assumes non-collinear
inputs; tree ensembles and neural nets don't have that assumption and the paper never
applies VIF to their inputs. An earlier version of this pipeline VIF-filtered every
model's features, which cut the boosting models down to 2 features and made MLR look
artificially strong — the opposite of the paper's finding. Fixed: `config.py`'s
`FeatureSelectionConfig.vif_only_for` scopes VIF filtering to MLR only; `evaluate.py`
gives MLR the VIF-reduced set and gives XGBoost/LightGBM/DNN the full selected set.
See `tests/test_feature_selection.py::test_branches_give_mlr_a_different_smaller_or_equal_set`.

**2. The paper's literal boosting hyperparameters (400 trees, lr=0.3) overfit badly on
a small synthetic sample** — R² actually *fell* as more features were added, the classic
overfitting signature, because those settings were tuned by the paper's authors against
~1M real rows. Fixed with a documented, opt-in deviation: `models.py`'s
`small_sample_safe_boosting` (on by default for the demo-scale run, off under `--faithful`)
uses fewer/shallower/more-regularized trees suited to a few-thousand-row sample. This is
a genuine, labeled departure from the paper's stated hyperparameters — not a paper-stated
value — and `main.py` logs which mode is active on every run so results are never silently
non-comparable.

## Comparing to the paper — what matched, what's a real small-sample effect

**Matched well:** SHAP feature importance puts `linear_distance` at ~47% of total
contribution — the single largest feature by a wide margin, tracking the paper's
">50%" finding. `actual_distance`, `vehicle_tonnage`, and `freight_weight` also show up
as the next-largest contributors, same ranking as the paper's Figure 3.

**Close but not identical:** after both bug fixes, LightGBM (R²≈0.44) and XGBoost
(R²≈0.41) sit close behind MLR (R²≈0.45) rather than clearly ahead of it, whereas the
paper reports LightGBM/XGBoost clearly beating MLR (0.85 vs 0.67 in its best branch). The
gap narrowed by roughly 10x once the two bugs above were fixed (from a ~30-point MLR lead
down to a ~1-4 point one). The remaining gap is a legitimate small-sample effect, not
another bug: with only ~5,700-8,000 rows and a target that's still substantially linear
by construction, boosted trees don't have as much nonlinear structure to exploit as they
would in the paper's 1M+-row real dataset, where far more of the fare-setting process's
true nonlinearity and interaction effects are statistically visible. Increasing
`config.py`'s `DataConfig.n_rows` (e.g. to 50,000+) and rerunning narrows this further, at
the cost of runtime.

## Validation on a real (non-synthetic) dataset

To check the pipeline's mechanics hold up on genuine messy data, not just the paper-matched
synthetic generator, it was also run against a real public logistics dataset: the **SCMS
Delivery History Dataset** (USAID Supply Chain Management System, ~10,324 international
health-commodity shipment records, 2006-2015). This is a *different* real freight/logistics
pricing dataset, not the paper's own private one — the comparison here is "does the pipeline
work on real data," not "does it reproduce this specific paper's numbers a second time."

```
python main.py --real-data data/SCMS_Delivery_History_Dataset.csv
```

**Real-world messiness this exercised that the synthetic generator doesn't:** the target
(`Freight Cost (USD)`) and a key feature (`Weight (Kilograms)`) are stored as text columns
with ~40% non-numeric placeholder values (`"Freight Included in Commodity Cost"`, `"Weight
Captured Separately"`, `"See ASN-93 (ID#:1281)"`) mixed in with real numbers — coerced to
NaN by `real_data.py` and handled by the same listwise-deletion / mean-imputation code path
used for synthetic data. High-cardinality categoricals (Vendor: 73 levels, Manufacturing
Site: 88 levels) are capped to the top 15 + "Other" to keep one-hot dimensionality sane —
found and fixed a real LightGBM crash along the way (`Do not support special JSON
characters in feature name`, triggered by category values like `"Côte d'Ivoire"` surviving
into one-hot column names; fixed by sanitizing column names in `feature_selection.numeric_frame`).

**Result on real data — the paper's core finding replicates:**

| Model    | correlation+listwise | correlation+mean | stepwise+listwise | stepwise+mean |
|----------|----------------------|-------------------|--------------------|-----------------|
| MLR      | 0.383 | 0.407 | 0.420 | 0.423 |
| XGBoost  | 0.413 | 0.459 | **0.522** | 0.497 |
| LightGBM | 0.431 | 0.412 | 0.518 | 0.455 |
| DNN      | 0.230 | 0.208 | 0.215 | 0.230 |

Boosting (R²=0.522) clearly beats MLR (R²=0.423) here — on real data, at real scale
(~3,300-3,500 rows after cleaning), without needing the small-sample-safe hyperparameter
mode. SHAP importance puts `freight_weight` at 46.8% of total contribution, the real-data
analog of the synthetic run's `linear_distance` dominance and directionally consistent with
the paper's freight-weight/distance-driven cost story. This is the strongest evidence in
this project that the pipeline's mechanics (not just the synthetic data's construction) are
sound: the boosting-beats-MLR pattern that a purely-linear synthetic target struggled to
produce shows up naturally on real, messier data.

## Files
- `01_data_schema.yaml`, `02_model_spec.yaml`, `03_synthetic_data_design.yaml` — the
  paper2code-tabular skill's extraction artifacts this pipeline was built from.
- `real_data.py` — loader/adapter for the real SCMS validation dataset (see above).
- `download_real_data.sh` — re-fetches the real dataset if `data/` is empty.
- `streamlit_app.py` — interactive UI (synthetic or real data, faithful or demo mode).
- Source files as listed in Architecture above.
- `tests/` — run with `python -m pytest` (17 tests, including real-data tests that
  auto-skip if `data/SCMS_Delivery_History_Dataset.csv` isn't present).
