# Generality test: paper2code-tabular on a second paper

To check the skill's phases generalize beyond the freight-cost paper they were built
against, I ran Phase 1 (data schema extraction) and Phase 2 (model spec extraction)
against a structurally different tabular ML paper:

**Sharma, Harsora & Ogunleye (2024), "An Optimal House Price Prediction Algorithm:
XGBoost," Analytics 3(1):30-45.** https://doi.org/10.3390/analytics3010003

Chosen deliberately for its differences from the freight paper: different domain (real
estate vs. logistics), different metric set (R², adjusted R², MSE, RMSE, MAE, CV score
vs. just R²), a paper whose central contribution is hyperparameter tuning itself
(GridSearchCV, untuned vs. tuned) rather than a fixed "basic form" per model, feature
importance via XGBoost's native method rather than SHAP, and — critically — a paper that
discloses far less preprocessing/selection specificity than the freight paper did.

## What generalized fine
Phase 1's core schema fields (dataset facts, target variable, kept/dropped variables)
and Phase 2's core fields (per-model hyperparameters, metrics, feature importance)
mapped cleanly. The YAML structure itself didn't need to change.

## What didn't generalize without adjustment — two real findings

1. **Papers vary enormously in disclosed preprocessing detail.** The freight paper
   published explicit IQR thresholds-by-implication, exact row counts per missing-data
   variant, and full Pearson-r/VIF tables. This paper names preprocessing *categories*
   ("data cleaning," "denoising," "outlier handling") without saying which technique was
   applied to which column, and names its feature-selection method (random forest) without
   publishing the selected-variable list. Phase 1's `open_questions` field ends up doing
   most of the work for a paper like this — which is correct behavior, but the skill's
   instructions didn't originally call this out as an expected mode, so I added guidance
   (see below) rather than let a future run either invent false specifics or stall.

2. **A paper's contribution can BE the hyperparameter tuning.** The freight paper reports
   one hyperparameter set per model. This paper's whole point is comparing untuned vs.
   GridSearchCV-tuned versions of five models — meaning "hyperparameters" isn't a single
   block to extract per model, it's two. The original Phase 2 template didn't anticipate this.

## What I changed in the skill as a result

Added guidance to both `references/01_data_schema_extraction.md` and
`references/02_model_spec_extraction.md` telling future runs to: (a) let sparse papers
produce sparse, honest YAMLs rather than padded ones, and (b) check whether a paper's
hyperparameters are a single fixed set or an untuned-vs-tuned comparison before assuming
the freight paper's pattern. This is a real fix based on real friction, not a speculative one.

## What I did NOT do (scope of this test)
Only Phases 1-2 were run for this second paper — not Phase 3 (synthetic data design) or
Phase 4 (full pipeline implementation). Building a second complete runnable pipeline was
out of scope for a generality check; the goal here was confirming the extraction phases
themselves hold up on a differently-structured paper, which they did once the two
adjustments above were made.
