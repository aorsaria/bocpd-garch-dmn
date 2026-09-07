"""Configuration loading for changepoint experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration of a BOCPD experiment profile.

    Attributes:
        name: Profile identifier used in output paths.
        full_scale: Whether the profile is the dissertation-scale experiment.
        seed: Root deterministic random seed.
        particles: Particles retained for each run-length hypothesis.
        max_run_lengths: Maximum number of run-length hypotheses retained.
        hazard: Constant prior changepoint probability per observation.
        young_window: Run-length cutoff used for the recent-regime score.
        burn_in: Initial observations excluded from detector evaluation.
        detection_window: Maximum post-change delay counted as a detection.
        target_arl0: Target in-control average run length.
        null_calibration_streams: Null streams used to calibrate thresholds.
        null_validation_streams: Independent null streams used to estimate ARL0.
        null_horizon: Number of observations in each null stream.
        single_change_trials: Replications for each single-change scenario.
        single_change_time: True changepoint position in single-change streams.
        single_change_horizon: Length of each single-change stream.
        monitoring_streams: Replications in continuous monitoring.
        monitoring_horizon: Length of each continuous-monitoring stream.
        minimum_regime_duration: Minimum simulated regime duration.
        real_calibration_observations: Pre-2007 observations used per contract
            for market threshold calibration.
        selection_observations: Pre-2007 observations used for innovation-law
            selection.
        selection_seeds: Particle-filter seeds averaged for innovation selection.
        garch_fit_starts: Optimisation starts per GARCH(1,1) fit.
        representative_tickers: Contracts retained for representative outputs.
    """

    name: str
    full_scale: bool
    seed: int
    particles: int
    max_run_lengths: int
    hazard: float
    young_window: int
    burn_in: int
    detection_window: int
    target_arl0: int
    null_calibration_streams: int
    null_validation_streams: int
    null_horizon: int
    single_change_trials: int
    single_change_time: int
    single_change_horizon: int
    monitoring_streams: int
    monitoring_horizon: int
    minimum_regime_duration: int
    real_calibration_observations: int
    selection_observations: int
    selection_seeds: int
    garch_fit_starts: int
    representative_tickers: tuple[str, ...]

    @classmethod
    def from_toml(cls, path: Path) -> "ExperimentConfig":
        """Load a profile from ``path`` and normalise sequence-valued fields."""
        values = tomllib.loads(path.read_text())
        values["representative_tickers"] = tuple(values["representative_tickers"])
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping of all profile parameters."""
        return asdict(self)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_ROOT = PACKAGE_ROOT / "profiles"
DEFAULT_ARCHIVE = PROJECT_ROOT / "pinnacle.zip"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "changepoint"
DEFAULT_BUILD_ROOT = PROJECT_ROOT / "build" / "changepoint"


def load_profile(name: str) -> ExperimentConfig:
    """Load a bundled BOCPD profile by name.

    Args:
        name: Profile stem, currently ``"smoke"`` or ``"full"``.

    Returns:
        Parsed immutable experiment configuration.

    Raises:
        KeyError: If no bundled profile has the requested name.
    """
    path = CONFIG_ROOT / f"{name}.toml"
    if not path.exists():
        raise KeyError(f"unknown experiment profile: {name}")
    return ExperimentConfig.from_toml(path)
