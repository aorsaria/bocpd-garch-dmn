"""Particle-filter implementation of BOCPD and BOCPD-GARCH."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .models import ObservationModel, RegimePrior


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Compute a numerically stable log-sum-exp reduction.

    Args:
        values: Log-scale input array.
        axis: Reduction axis, or ``None`` to reduce all entries.

    Returns:
        Scalar or array of reduced log sums.
    """
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    if axis is None:
        return float(result.reshape(()))
    return np.squeeze(result, axis=axis)


def systematic_resample(
    rng: np.random.Generator, weights: np.ndarray, count: int
) -> np.ndarray:
    """Draw indices by systematic resampling.

    Args:
        rng: NumPy random generator.
        weights: Nonnegative unnormalised particle weights.
        count: Number of indices to draw.

    Returns:
        Integer indices into ``weights``.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("resampling weights must be a nonempty vector")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0 or np.any(weights < 0):
        raise ValueError("resampling weights must be finite and nonnegative")
    cumulative = np.cumsum(weights / total)
    cumulative[-1] = 1.0
    positions = (rng.random() + np.arange(count)) / count
    return np.searchsorted(cumulative, positions, side="left")


@dataclass(frozen=True)
class DetectorStep:
    """Posterior summaries produced for one observation.

    Attributes:
        cp_prob: Posterior probability of run length zero.
        young_prob: Posterior probability of run length at most ``young_window``.
        expected_run_length: Posterior mean run length.
        run_length_variance: Posterior variance of run length.
        expected_variance: Posterior mean conditional variance.
        expected_volatility: Posterior mean conditional standard deviation.
        predictive_log_likelihood: Log predictive density of the observation.
    """

    cp_prob: float
    young_prob: float
    expected_run_length: float
    run_length_variance: float
    expected_variance: float
    expected_volatility: float
    predictive_log_likelihood: float


class ParticleBOCPD:
    """BOCPD with static regime parameters represented by particles.

    A new regime always receives a complete conditional-variance reset to
    its sampled long-run variance. Parameters are drawn from ``prior`` and
    conditioned sequentially by importance weighting. No contagion hazard,
    volatility inheritance, or data-dependent new-regime proposal is used.
    """

    OUTPUT_COLUMNS = tuple(DetectorStep.__dataclass_fields__)

    def __init__(
        self,
        model: ObservationModel,
        prior: RegimePrior,
        *,
        particles: int = 500,
        hazard: float = 1.0 / 250.0,
        max_run_lengths: int = 250,
        ess_fraction: float = 0.5,
        prune_threshold: float = 1e-8,
        young_window: int = 30,
        seed: int = 0,
    ) -> None:
        """Initialise a particle approximation to the BOCPD recursion.

        Args:
            model: Within-regime observation model.
            prior: Prior used when a new regime is proposed.
            particles: Particle count for every retained run length.
            hazard: Constant changepoint probability per observation.
            max_run_lengths: Maximum retained run-length hypotheses.
            ess_fraction: Resampling threshold as a fraction of particle count.
            prune_threshold: Minimum posterior mass for retaining a run length.
            young_window: Inclusive run-length cutoff for ``young_prob``.
            seed: Initial random seed.
        """
        if particles < 2 or max_run_lengths < 1:
            raise ValueError("particles must be >= 2 and max_run_lengths >= 1")
        if not (0 < hazard < 1):
            raise ValueError("hazard must lie strictly between zero and one")
        if not (0 < ess_fraction <= 1) or not (0 <= prune_threshold < 1):
            raise ValueError("invalid resampling or pruning setting")
        if young_window < 0:
            raise ValueError("young_window must be nonnegative")
        self.model = model
        self.prior = prior
        self.particles = int(particles)
        self.hazard = float(hazard)
        self.max_run_lengths = int(max_run_lengths)
        self.ess_fraction = float(ess_fraction)
        self.prune_threshold = float(prune_threshold)
        self.young_window = int(young_window)
        self._initial_seed = int(seed)
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """Clear posterior state and reset the random generator.

        Args:
            seed: Replacement seed, or ``None`` to reuse the initial seed.
        """
        self.rng = np.random.default_rng(self._initial_seed if seed is None else seed)
        self.initialized = False
        self.mu = self.omega = self.alpha = self.beta = None
        self.variance = self.residual = self.log_weights = None
        self.log_joint = None
        self.run_lengths = None
        self.max_pooling_error = 0.0

    def _new_parameters(self, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
        """Draw new-regime parameters with the requested particle shape."""
        return self.prior.sample(self.rng, shape, self.model.uses_garch)

    def _first_update(self, observation: float) -> DetectorStep:
        """Initialise the posterior from the first observation."""
        proposal = self._new_parameters((1, self.particles))
        likelihood = self.model.log_likelihood(
            observation, proposal["mu"], proposal["hbar"]
        )
        predictive = float(logsumexp(likelihood) - np.log(self.particles))
        log_weights = likelihood - float(logsumexp(likelihood))
        self.mu = proposal["mu"]
        self.omega = proposal["omega"]
        self.alpha = proposal["alpha"]
        self.beta = proposal["beta"]
        self.variance = proposal["hbar"]
        self.residual = observation - self.mu
        self.log_weights = log_weights
        self.log_joint = np.array([0.0])
        self.run_lengths = np.array([0], dtype=int)
        self.initialized = True
        self._resample_degenerate_rows()
        return self._summarise(predictive)

    def update(self, observation: float) -> DetectorStep:
        """Assimilate one observation and return posterior summaries.

        Args:
            observation: Finite scalar observation.

        Returns:
            Current changepoint, run-length, variance, and predictive summaries.
        """
        observation = float(observation)
        if not np.isfinite(observation):
            raise ValueError("observations must be finite")
        if not self.initialized:
            return self._first_update(observation)

        log_hazard = np.log(self.hazard)
        log_survival = np.log1p(-self.hazard)
        predicted_variance = (
            self.omega
            + self.alpha * self.residual**2
            + self.beta * self.variance
        )
        np.clip(predicted_variance, 1e-12, 1e8, out=predicted_variance)
        continuation_ll = self.model.log_likelihood(
            observation, self.mu, predicted_variance
        )
        continuation_evidence = logsumexp(
            self.log_weights + continuation_ll, axis=1
        )

        proposal = self._new_parameters(self.mu.shape)
        initial_variance = proposal["hbar"]
        changepoint_ll = self.model.log_likelihood(
            observation, proposal["mu"], initial_variance
        )
        changepoint_evidence = logsumexp(
            self.log_weights + changepoint_ll, axis=1
        )

        grown_joint = self.log_joint + log_survival + continuation_evidence
        new_joint = float(logsumexp(self.log_joint + log_hazard + changepoint_evidence))
        candidate_log_mass = (
            (self.log_joint + log_hazard)[:, None]
            + self.log_weights
            + changepoint_ll
        )
        self.max_pooling_error = max(
            self.max_pooling_error,
            abs(float(logsumexp(candidate_log_mass)) - new_joint),
        )
        candidate_weights = np.exp(candidate_log_mass - new_joint).ravel()
        candidate_indices = systematic_resample(
            self.rng, candidate_weights, self.particles
        )
        mean0 = proposal["mu"].ravel()[candidate_indices]
        row0 = {
            "mu": mean0,
            "omega": proposal["omega"].ravel()[candidate_indices],
            "alpha": proposal["alpha"].ravel()[candidate_indices],
            "beta": proposal["beta"].ravel()[candidate_indices],
            "variance": initial_variance.ravel()[candidate_indices],
            "residual": observation - mean0,
            "log_weights": np.full(self.particles, -np.log(self.particles)),
        }

        continuation_weights = (
            self.log_weights + continuation_ll - continuation_evidence[:, None]
        )
        continuation_residual = observation - self.mu
        self.mu = np.vstack((row0["mu"], self.mu))
        self.omega = np.vstack((row0["omega"], self.omega))
        self.alpha = np.vstack((row0["alpha"], self.alpha))
        self.beta = np.vstack((row0["beta"], self.beta))
        self.variance = np.vstack((row0["variance"], predicted_variance))
        self.residual = np.vstack((row0["residual"], continuation_residual))
        self.log_weights = np.vstack((row0["log_weights"], continuation_weights))
        self.log_joint = np.concatenate(([new_joint], grown_joint))
        self.run_lengths = np.concatenate(([0], self.run_lengths + 1))

        predictive = float(logsumexp(self.log_joint))
        self.log_joint -= predictive
        self._resample_degenerate_rows()
        self._prune_run_lengths()
        return self._summarise(predictive)

    def _resample_degenerate_rows(self) -> None:
        """Systematically resample run-length rows with low effective sample size."""
        weights = np.exp(self.log_weights)
        effective_size = 1.0 / np.sum(weights**2, axis=1)
        for row in np.flatnonzero(effective_size < self.ess_fraction * self.particles):
            indices = systematic_resample(self.rng, weights[row], self.particles)
            for array in (
                self.mu,
                self.omega,
                self.alpha,
                self.beta,
                self.variance,
                self.residual,
            ):
                array[row] = array[row, indices]
            self.log_weights[row] = -np.log(self.particles)

    def _prune_run_lengths(self) -> None:
        """Remove negligible run lengths and enforce the configured hard cap."""
        posterior = np.exp(self.log_joint)
        eligible = set(np.flatnonzero(posterior >= self.prune_threshold).tolist())
        eligible.add(0)
        if len(eligible) > self.max_run_lengths:
            others = sorted(eligible - {0}, key=lambda index: posterior[index], reverse=True)
            selected = sorted([0, *others[: self.max_run_lengths - 1]])
        else:
            selected = sorted(eligible)
        selected_array = np.asarray(selected, dtype=int)
        for name in (
            "mu",
            "omega",
            "alpha",
            "beta",
            "variance",
            "residual",
            "log_weights",
        ):
            setattr(self, name, getattr(self, name)[selected_array])
        self.log_joint = self.log_joint[selected_array]
        self.log_joint -= float(logsumexp(self.log_joint))
        self.run_lengths = self.run_lengths[selected_array]

    def _summarise(self, predictive: float) -> DetectorStep:
        """Reduce the particle/run-length posterior to scalar diagnostics."""
        run_posterior = np.exp(self.log_joint)
        particle_weights = np.exp(self.log_weights)
        expected_run_length = float(run_posterior @ self.run_lengths)
        expected_variance = float(
            run_posterior @ np.sum(particle_weights * self.variance, axis=1)
        )
        expected_volatility = float(
            run_posterior
            @ np.sum(particle_weights * np.sqrt(self.variance), axis=1)
        )
        cp_prob = float(run_posterior[self.run_lengths == 0].sum())
        young_prob = float(
            run_posterior[self.run_lengths <= self.young_window].sum()
        )
        return DetectorStep(
            cp_prob=float(np.clip(cp_prob, 0.0, 1.0)),
            young_prob=float(np.clip(young_prob, 0.0, 1.0)),
            expected_run_length=expected_run_length,
            run_length_variance=float(
                run_posterior @ (self.run_lengths - expected_run_length) ** 2
            ),
            expected_variance=expected_variance,
            expected_volatility=expected_volatility,
            predictive_log_likelihood=float(predictive),
        )

    def run(
        self,
        values: np.ndarray | pd.Series,
        *,
        store_posterior: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
        """Run the detector over a complete observation sequence.

        Args:
            values: NumPy vector or indexed pandas series of observations.
            store_posterior: Whether to retain run-length support and mass at
                every observation.

        Returns:
            Data frame of posterior summaries, optionally paired with the full
            run-length posterior history.
        """
        index = values.index if isinstance(values, pd.Series) else pd.RangeIndex(len(values))
        array = np.asarray(values, dtype=float)
        rows: list[dict[str, float]] = []
        posterior = []
        for observation in array:
            rows.append(asdict(self.update(float(observation))))
            if store_posterior:
                posterior.append((self.run_lengths.copy(), np.exp(self.log_joint)))
        frame = pd.DataFrame(rows, index=index)
        if store_posterior:
            return frame, posterior
        return frame


def make_detector(
    name: str,
    prior: RegimePrior,
    *,
    nu: float | None = None,
    **settings: object,
) -> ParticleBOCPD:
    """Construct a configured detector by experiment name.

    Args:
        name: ``bocpd``, ``bocpd-garch``, or ``bocpd-garch-t``.
        prior: New-regime parameter prior.
        nu: Student-t degrees of freedom for ``bocpd-garch-t``.
        **settings: Keyword arguments forwarded to :class:`ParticleBOCPD`.

    Returns:
        Configured particle detector.
    """
    models = {
        "bocpd": ObservationModel("iid-gaussian"),
        "bocpd-garch": ObservationModel("garch-gaussian"),
    }
    normalised = name.lower()
    if normalised == "bocpd-garch-t":
        if nu is None:
            raise ValueError("bocpd-garch-t requires nu")
        model = ObservationModel("garch-student-t", float(nu))
    elif normalised in models:
        if nu is not None:
            raise ValueError(f"{normalised} does not accept nu")
        model = models[normalised]
    else:
        raise KeyError(f"unknown detector: {name}")
    return ParticleBOCPD(model, prior, **settings)
