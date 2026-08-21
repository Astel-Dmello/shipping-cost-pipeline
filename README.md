# Freight Shipping-Cost Prediction — Paper-to-Pipeline

A reproduction of [Jang, Chang & Kim (2023)](https://doi.org/10.3390/su15021122), *Prediction
of Shipping Cost on Freight Brokerage Platform Using Machine Learning* (Sustainability),
built with a custom Claude Code skill I wrote (`paper2code-tabular`) that converts tabular
ML papers into runnable pipelines.

**[Live demo →](#)** https://shipping-cost-pipeline-djwdlnjfiwf9arj74urmbn.streamlit.app/

## What this is

The paper compares four models (linear regression, a deep neural net, XGBoost, LightGBM)
for predicting freight shipping costs from cargo/vehicle/route features. Its dataset is
private, so I built two ways to validate the pipeline instead of one:

1. **Synthetic data** generated to match the paper's described schema and reported
   correlations.
2. **A real public logistics dataset** (USAID's Supply Chain Management System shipment
   history, ~10K rows) to confirm the pipeline's mechanics hold up on genuinely messy,
   real-world data — not just data I built to be convenient.

## What I found

Building this surfaced two real bugs, not just "the numbers didn't match":

- **A methodology-scoping bug.** The paper's VIF multicollinearity filter applies to its
  linear regression model only — not to the tree/neural models, which don't share OLS's
  non-collinearity assumption. My first pass applied it everywhere, which quietly starved
  the boosting models of features and made linear regression look artificially strong —
  the opposite of what the paper found. Traced it, fixed the scoping, added a regression
  test so it can't silently reappear.
- **A data-scale mismatch.** The paper's boosting hyperparameters (400 trees, aggressive
  learning rate) were tuned against 1M+ real rows. Applied literally to a smaller synthetic
  sample, they overfit — R² actually *fell* as more features were added. Fixed with a
  documented, opt-in "small-sample-safe" hyperparameter mode.

**On the real dataset, the paper's core finding replicates cleanly**: boosting models beat
linear regression (XGBoost R²=0.52 vs. MLR R²=0.42), and the dominant predictive feature
(cargo weight, this dataset's analog to the paper's "linear distance") matches the paper's
SHAP-importance story.

## Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · TensorFlow/Keras · SHAP · Streamlit ·
pytest (17 tests, including a regression test tied to the VIF bug above)

## Run it locally

```bash
git clone <this-repo>
cd shipping_cost_pipeline
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or the CLI: `python main.py` (synthetic data) / `python main.py --real-data data/SCMS_Delivery_History_Dataset.csv` (real data).

See [TECHNICAL.md](TECHNICAL.md) for the full technical writeup — architecture, the exact bug
diagnoses, and what did/didn't match the paper — and [DEPLOY.md](DEPLOY.md) for deployment
instructions.

## Why this exists

Built as a test case for `paper2code-tabular`, a Claude Code skill I wrote as a companion
to Anthropic's `paper2code` skill — generic paper2code targets deep-learning architecture
papers; this one targets tabular ML papers, where the hard part is reconstructing a data
pipeline and multi-model comparison rather than implementing a novel architecture.
