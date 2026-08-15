"""
Centralized configuration. Every tunable knob in the pipeline lives here so main.py's
CLI flags, tests, and the individual modules all read from one source of truth instead
of scattering magic numbers across files.

Two groups of settings:
  - PAPER_* : the protocol values actually reported in Jang, Chang & Kim (2023).
  - DEMO_*  : this pipeline's runtime-friendly defaults for a synthetic, laptop-scale run.
Defaults below use the DEMO values; pass --faithful on the CLI (see main.py) to switch
to the PAPER values (much slower, especially the 500-epoch DNN x 30 folds).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    n_rows: int = 8000          # paper: 1,885,033 (demo-scale for runtime, see README)
    seed: int = 42


@dataclass(frozen=True)
class PreprocessConfig:
    iqr_multiplier: float = 1.5   # paper doesn't state its multiplier; 1.5 is the standard default


@dataclass(frozen=True)
class FeatureSelectionConfig:
    correlation_threshold: float = 0.1     # paper Sec 4.1, stated explicitly
    stepwise_n_features: int = 15          # paper's stepwise branch selects 33-35; keep smaller for runtime
    vif_threshold: float = 10.0            # paper Sec 5.2, stated explicitly
    # IMPORTANT: the paper only runs VIF-based multicollinearity removal for the MLR
    # model (Sec 5.2, "MLR"). Tree ensembles and the DNN are not sensitive to correlated
    # inputs the way OLS is, and the paper never applies VIF filtering to their inputs.
    # Applying VIF everywhere (an earlier version of this pipeline did) silently starves
    # the boosting models of signal and is why they underperformed MLR in that version --
    # see 03_synthetic_data_design.yaml fidelity_notes for the writeup.
    vif_only_for: tuple = ("MLR",)


@dataclass(frozen=True)
class ValidationConfig:
    test_size: float = 0.2
    n_folds: int = 5              # paper: 30. Demo default trades fidelity for runtime.
    dnn_epochs: int = 30          # paper: 500. Demo default trades fidelity for runtime.
    confidence: float = 0.95
    random_state: int = 42
    # See models.py build_xgboost/build_lightgbm docstrings: the paper's literal boosting
    # hyperparameters (400 trees, lr=0.3) overfit badly on a small synthetic sample.
    # True by default here since this pipeline's default n_rows is demo-scale; set False
    # (or use --faithful, which forces this False) to use the paper's exact hyperparameters.
    small_sample_safe_boosting: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig = DataConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    feature_selection: FeatureSelectionConfig = FeatureSelectionConfig()
    validation: ValidationConfig = ValidationConfig()


FAITHFUL_VALIDATION = ValidationConfig(n_folds=30, dnn_epochs=500, small_sample_safe_boosting=False)


def get_config(faithful: bool = False) -> PipelineConfig:
    """Returns the demo config by default, or the paper's actual protocol values
    (much slower to run) when faithful=True."""
    if faithful:
        return PipelineConfig(validation=FAITHFUL_VALIDATION)
    return PipelineConfig()
