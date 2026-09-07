"""Configuration and model registry for the deep-momentum experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib

from experiment_data import (
    END_DATE,
    FIRST_TEST_START,
    HISTORY_START,
    STUDY_START,
)


WINDOWS = (1995, 2000, 2005, 2010, 2015, 2020)
PRIMARY_WINDOWS = WINDOWS
TEST_SPAN = 5
SEQUENCE_LENGTH = 63
VOLATILITY_TARGET = 0.15
RETURN_HORIZONS = (1, 21, 63, 126, 252)
MACD_PAIRS = ((8, 24), (16, 48), (32, 96))
BASE_FEATURES = tuple(f"norm_ret_{h}" for h in RETURN_HORIZONS) + tuple(
    f"macd_{short}_{long}" for short, long in MACD_PAIRS
)


@dataclass(frozen=True)
class ModelSpec:
    """Definition of one reported deep-momentum model.

    Attributes:
        name: Stable model label used in files and result tables.
        family: Feature-family identifier used to select tensor inputs.
        feature_columns: Ordered model input columns.
        cost_bps: Transaction-cost rate embedded in the training objective.
    """

    name: str
    family: str
    feature_columns: tuple[str, ...]
    cost_bps: float = 0.0


def _spec(
    name: str, family: str, additions: tuple[str, ...], cost_bps: float = 0.0
) -> ModelSpec:
    """Construct a model specification by extending the eight base features."""
    return ModelSpec(name, family, BASE_FEATURES + additions, cost_bps)


_GROSS_SPECS = (
    _spec("LSTM", "base", ()),
    _spec("LSTM-BOCPD", "bocpd", ("bocpd_score", "bocpd_rl")),
    _spec(
        "LSTM-BOCPD-rich",
        "bocpd_rich",
        ("bocpd_score", "bocpd_rl", "bocpd_cp_prob", "bocpd_exp_vol"),
    ),
    _spec(
        "LSTM-BOCPD-sys",
        "bocpd_sys",
        ("bocpd_score", "bocpd_rl", "sys_score", "sys_frac"),
    ),
    _spec(
        "LSTM-BOCPD-full",
        "bocpd_full",
        (
            "bocpd_score",
            "bocpd_rl",
            "bocpd_cp_prob",
            "bocpd_exp_vol",
            "sys_score",
            "sys_frac",
        ),
    ),
    _spec("LSTM-CUSUM-mv", "cusum", ("cmv_stat", "cmv_recent")),
)


def _cost_clone(base: ModelSpec) -> ModelSpec:
    """Return the reported two-basis-point objective variant of a gross model."""
    return ModelSpec(
        name=f"{base.name}@2bp",
        family=base.family,
        feature_columns=base.feature_columns,
        cost_bps=2.0,
    )


_COST_BASE_NAMES = {"LSTM", "LSTM-BOCPD", "LSTM-BOCPD-rich"}
_COST_BASES = tuple(spec for spec in _GROSS_SPECS if spec.name in _COST_BASE_NAMES)
MODEL_SPECS = _GROSS_SPECS + tuple(_cost_clone(spec) for spec in _COST_BASES)
MODEL_BY_NAME = {spec.name: spec for spec in MODEL_SPECS}
MODEL_NAMES = tuple(MODEL_BY_NAME)
FEATURE_FAMILIES = tuple(dict.fromkeys(spec.family for spec in MODEL_SPECS))


@dataclass(frozen=True)
class Profile:
    """Complete deep-momentum experiment configuration.

    Attributes:
        name: Profile identifier used in cache and output paths.
        full_scale: Whether this is the dissertation-scale experiment.
        seed: Root deterministic random seed.
        windows: Out-of-sample test-window start years.
        test_span: Length of each test window in years.
        tickers: Optional contract subset; empty selects the full universe.
        data_start: First date eligible for estimation and evaluation.
        trials: Random-search configurations per model and window.
        ensemble: Maximum configured reporting ensemble size.
        replicates: Seed replications of each selected configuration.
        max_epochs: Maximum training epochs per fit.
        patience: Early-stopping patience in epochs.
        particles: BOCPD particles per retained run length.
        max_run_lengths: Maximum retained BOCPD run-length hypotheses.
        hazard: Constant BOCPD changepoint probability per observation.
        young_window: Run-length cutoff for the recent-regime probability.
        selection_observations: Training observations used for innovation selection.
        selection_burn_in: Initial selection observations excluded from scoring.
        selection_seeds: Particle-filter seeds averaged for innovation selection.
        garch_fit_starts: Optimisation starts per contract GARCH fit.
        minimum_prior_observations: Minimum observations required for a GARCH fit.
        chart_calibration_observations: Training observations used for CUSUM
            threshold calibration.
        chart_target_arl: Target average run length of the CUSUM comparator.
        chart_burn_in: Initial observations used to estimate CUSUM moments.
        bootstrap_samples: Paired circular-block bootstrap replications.
        history_start: Optional pre-study start used only for indicator warm-up.
    """

    name: str
    full_scale: bool
    seed: int
    windows: tuple[int, ...]
    test_span: int
    tickers: tuple[str, ...]
    data_start: str
    trials: int
    ensemble: int
    replicates: int
    max_epochs: int
    patience: int
    particles: int
    max_run_lengths: int
    hazard: float
    young_window: int
    selection_observations: int
    selection_burn_in: int
    selection_seeds: int
    garch_fit_starts: int
    minimum_prior_observations: int
    chart_calibration_observations: int
    chart_target_arl: int
    chart_burn_in: int
    bootstrap_samples: int
    # Pre-study warm-up start (indicator initialisation only). Cache-neutral:
    # excluded from to_dict() so detector and tensor recipes are unchanged; it
    # enters the base-feature fingerprint explicitly when set.
    history_start: str | None = None

    @classmethod
    def from_toml(cls, path: Path) -> "Profile":
        """Load a profile from TOML and normalise sequence-valued fields."""
        values = tomllib.loads(path.read_text())
        values["windows"] = tuple(int(value) for value in values["windows"])
        values["tickers"] = tuple(values.get("tickers", ()))
        values.setdefault("history_start", HISTORY_START)
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        """Return cache-relevant parameters as a serialisable mapping."""
        values = asdict(self)
        del values["history_start"]  # cache-neutral, see field comment
        return values


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
PROFILE_ROOT = PACKAGE_ROOT / "profiles"
DEFAULT_ARCHIVE = PROJECT_ROOT / "pinnacle.zip"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "deep_momentum"
DEFAULT_BUILD_ROOT = PROJECT_ROOT / "build" / "deep_momentum"


def load_profile(name: str) -> Profile:
    """Load and validate a bundled deep-momentum profile.

    Args:
        name: Profile stem, currently ``"smoke"`` or ``"full"``.

    Returns:
        Parsed immutable experiment profile.
    """
    path = PROFILE_ROOT / f"{name}.toml"
    if not path.exists():
        raise KeyError(f"unknown deep-momentum profile: {name}")
    profile = Profile.from_toml(path)
    if profile.ensemble > profile.trials:
        raise ValueError("ensemble size cannot exceed the trial budget")
    if profile.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if profile.history_start is None:
        raise ValueError("history_start must be defined for indicator initialisation")
    if not (
        profile.history_start == HISTORY_START
        and profile.data_start >= STUDY_START
        and min(profile.windows) >= int(FIRST_TEST_START[:4])
    ):
        raise ValueError("profile violates the history/study/test date policy")
    return profile
