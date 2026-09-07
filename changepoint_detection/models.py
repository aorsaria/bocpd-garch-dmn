"""Observation models, regime priors, and GARCH QMLE utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln, logit

LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class ObservationModel:
    """Within-regime observation model used by the particle filter.

    Attributes:
        kind: IID Gaussian, Gaussian GARCH, or standardised Student-t GARCH law.
        nu: Student-t degrees of freedom; required only for Student-t GARCH.
    """

    kind: Literal["iid-gaussian", "garch-gaussian", "garch-student-t"]
    nu: float | None = None

    def __post_init__(self) -> None:
        """Validate the model kind and innovation-law parameters."""
        if self.kind == "garch-student-t":
            if self.nu is None or not np.isfinite(self.nu) or self.nu <= 2:
                raise ValueError("Student-t degrees of freedom must be finite and > 2")
        elif self.nu is not None:
            raise ValueError("nu is only valid for garch-student-t")

    @property
    def uses_garch(self) -> bool:
        """Whether conditional variance follows a GARCH recursion."""
        return self.kind != "iid-gaussian"

    def log_likelihood(
        self, observation: float, mean: np.ndarray, variance: np.ndarray
    ) -> np.ndarray:
        """Evaluate particle-wise predictive log likelihoods.

        Args:
            observation: Current scalar observation.
            mean: Particle-specific conditional means.
            variance: Particle-specific positive conditional variances.

        Returns:
            Log likelihood for each particle.
        """
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
            raise FloatingPointError("predictive variances must be positive and finite")
        residual2 = (observation - mean) ** 2
        if self.kind != "garch-student-t":
            return -0.5 * (LOG_2PI + np.log(variance) + residual2 / variance)
        nu = float(self.nu)
        constant = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log((nu - 2.0) * np.pi)
        )
        return (
            constant
            - 0.5 * np.log(variance)
            - (nu + 1.0) / 2.0 * np.log1p(residual2 / ((nu - 2.0) * variance))
        )


@dataclass(frozen=True)
class RegimePrior:
    """Asset-class prior for regime-level mean and volatility parameters.

    Attributes:
        mu_mean: Centre of the Gaussian prior for the regime mean.
        mu_sd: Standard deviation of the regime-mean prior.
        log_hbar_mean: Centre of the Gaussian prior for log long-run variance.
        log_hbar_sd: Standard deviation of the log-variance prior.
        alpha_low: Lower bound of the uniform GARCH shock-coefficient prior.
        alpha_high: Upper bound of the GARCH shock-coefficient prior.
        persistence_low: Lower bound of the total-persistence prior.
        persistence_high: Upper bound of the total-persistence prior.
        hbar_cap: Upper bound applied to sampled long-run variance.
    """

    mu_mean: float
    mu_sd: float
    log_hbar_mean: float
    log_hbar_sd: float
    alpha_low: float
    alpha_high: float
    persistence_low: float
    persistence_high: float
    hbar_cap: float = 25.0

    def __post_init__(self) -> None:
        """Validate finite scales and stationary coefficient intervals."""
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("prior values must be finite")
        if self.mu_sd <= 0 or self.log_hbar_sd <= 0 or self.hbar_cap <= 0:
            raise ValueError("prior scales and hbar_cap must be positive")
        if not (0 <= self.alpha_low < self.alpha_high < 1):
            raise ValueError("invalid alpha prior interval")
        if not (0 < self.persistence_low < self.persistence_high < 1):
            raise ValueError("invalid persistence prior interval")
        if self.alpha_high >= self.persistence_low:
            raise ValueError("alpha prior must lie below the persistence prior")

    def sample(
        self, rng: np.random.Generator, size: int | tuple[int, ...], uses_garch: bool
    ) -> dict[str, np.ndarray]:
        """Draw independent regime parameters.

        Args:
            rng: NumPy random generator.
            size: Output shape or scalar number of draws.
            uses_garch: Whether to draw GARCH coefficients or IID variances.

        Returns:
            Arrays for ``mu``, ``omega``, ``alpha``, ``beta``, and ``hbar``.
        """
        mean = rng.normal(self.mu_mean, self.mu_sd, size)
        hbar = np.minimum(
            np.exp(rng.normal(self.log_hbar_mean, self.log_hbar_sd, size)),
            self.hbar_cap,
        )
        if uses_garch:
            alpha = rng.uniform(self.alpha_low, self.alpha_high, size)
            persistence = rng.uniform(
                self.persistence_low, self.persistence_high, size
            )
            beta = persistence - alpha
            omega = hbar * (1.0 - persistence)
        else:
            alpha = np.zeros(size)
            beta = np.zeros(size)
            omega = hbar.copy()
        return {
            "mu": np.asarray(mean),
            "omega": np.asarray(omega),
            "alpha": np.asarray(alpha),
            "beta": np.asarray(beta),
            "hbar": np.asarray(hbar),
        }

    def to_dict(self) -> dict[str, float]:
        """Return a serialisable mapping of all prior parameters."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "RegimePrior":
        """Construct a prior from a mapping produced by :meth:`to_dict`."""
        return cls(**values)


SYNTHETIC_PRIOR = RegimePrior(
    mu_mean=0.0,
    mu_sd=0.5,
    log_hbar_mean=0.0,
    log_hbar_sd=0.75,
    alpha_low=0.02,
    alpha_high=0.25,
    persistence_low=0.60,
    persistence_high=0.98,
    hbar_cap=25.0,
)


@dataclass(frozen=True)
class GarchFit:
    """Result of a constant-mean Gaussian GARCH(1,1) QMLE fit.

    Attributes:
        mean: Fitted constant conditional mean.
        omega: Fitted variance intercept.
        alpha: Fitted coefficient on the lagged squared residual.
        beta: Fitted coefficient on lagged conditional variance.
        hbar: Implied long-run conditional variance.
        persistence: Sum of ``alpha`` and ``beta``.
        negative_log_likelihood: Optimised Gaussian negative log likelihood.
        converged: Whether the optimiser reported successful convergence.
        iterations: Number of optimiser iterations.
    """

    mean: float
    omega: float
    alpha: float
    beta: float
    hbar: float
    persistence: float
    negative_log_likelihood: float
    converged: bool
    iterations: int

    def to_dict(self) -> dict[str, float | bool | int]:
        """Return a serialisable mapping of fitted parameters and diagnostics."""
        return asdict(self)

    @property
    def has_finite_interior_solution(self) -> bool:
        """Whether the fit is finite and inside the stationary parameter space."""
        values = np.asarray(
            (
                self.mean,
                self.omega,
                self.alpha,
                self.beta,
                self.hbar,
                self.persistence,
                self.negative_log_likelihood,
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            return False
        return bool(
            self.hbar > 0
            and self.omega > 0
            and 0 < self.alpha < self.persistence < 0.999
            and self.beta > 0
            and np.isclose(self.alpha + self.beta, self.persistence)
        )


def _garch_parameters(unconstrained: np.ndarray) -> tuple[float, ...]:
    """Map unconstrained optimiser variables into stationary GARCH parameters."""
    mean = float(unconstrained[0])
    hbar = float(np.exp(unconstrained[1]))
    persistence = float(0.999 * expit(unconstrained[2]))
    alpha = float(persistence * expit(unconstrained[3]))
    beta = persistence - alpha
    omega = hbar * (1.0 - persistence)
    return mean, omega, alpha, beta, hbar, persistence


def _garch_nll(unconstrained: np.ndarray, values: np.ndarray) -> float:
    """Evaluate the Gaussian GARCH(1,1) negative log likelihood."""
    mean, omega, alpha, beta, _, _ = _garch_parameters(unconstrained)
    residual = values - mean
    variance = np.empty(len(values), dtype=float)
    variance[0] = max(float(np.var(values)), 1e-6)
    for index in range(1, len(values)):
        variance[index] = (
            omega + alpha * residual[index - 1] ** 2 + beta * variance[index - 1]
        )
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        return 1e100
    return float(0.5 * np.sum(LOG_2PI + np.log(variance) + residual**2 / variance))


def fit_garch11(
    values: np.ndarray, *, starts: int = 5, seed: int = 0
) -> GarchFit:
    """Fit stationary constant-mean GARCH(1,1) by Gaussian QMLE.

    Args:
        values: Finite, nonconstant, one-dimensional observations.
        starts: Number of deterministic/random optimisation initialisations.
        seed: Seed used to generate initialisations after the first.

    Returns:
        Best finite interior solution by negative log likelihood.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 100:
        raise ValueError("GARCH fitting requires a one-dimensional series of length >= 100")
    if np.any(~np.isfinite(values)) or float(np.var(values)) <= 0:
        raise ValueError("GARCH fitting values must be finite and nonconstant")
    if starts < 1:
        raise ValueError("starts must be positive")

    rng = np.random.default_rng(seed)
    variance = float(np.var(values))
    candidates: list[GarchFit] = []
    for index in range(starts):
        persistence = 0.90 if index == 0 else rng.uniform(0.75, 0.98)
        alpha_share = 0.08 / persistence if index == 0 else rng.uniform(0.03, 0.25)
        initial = np.array(
            [
                float(np.mean(values)),
                np.log(max(variance, 1e-6)),
                logit(np.clip(persistence / 0.999, 1e-5, 1 - 1e-5)),
                logit(np.clip(alpha_share, 1e-5, 1 - 1e-5)),
            ]
        )
        result = minimize(
            _garch_nll,
            initial,
            args=(values,),
            method="L-BFGS-B",
            bounds=((None, None), (-12, 12), (-12, 12), (-12, 12)),
            options={"maxiter": 3000, "ftol": 1e-11, "gtol": 1e-7},
        )
        mean, omega, alpha, beta, hbar, persistence = _garch_parameters(result.x)
        fit = GarchFit(
            mean=mean,
            omega=omega,
            alpha=alpha,
            beta=beta,
            hbar=hbar,
            persistence=persistence,
            negative_log_likelihood=float(result.fun),
            converged=bool(result.success and np.isfinite(result.fun)),
            iterations=int(result.nit),
        )
        if fit.has_finite_interior_solution:
            candidates.append(fit)
    if not candidates:
        raise RuntimeError("GARCH QMLE produced no finite interior solution")
    return min(candidates, key=lambda fit: fit.negative_log_likelihood)


def prior_from_fits(
    fits: Iterable[GarchFit],
    *,
    mu_sd_floor: float = 0.05,
    log_hbar_sd_floor: float = 0.25,
) -> RegimePrior:
    """Estimate a robust hierarchical regime prior from contract-level fits.

    Args:
        fits: At least two finite stationary GARCH fits from one asset class.
        mu_sd_floor: Minimum standard deviation for the regime-mean prior.
        log_hbar_sd_floor: Minimum standard deviation for log long-run variance.

    Returns:
        Empirical asset-class regime prior.
    """
    fits = tuple(fits)
    if len(fits) < 2:
        raise ValueError("at least two GARCH fits are required for a class prior")
    if any(not fit.has_finite_interior_solution for fit in fits):
        raise ValueError("class prior cannot include invalid GARCH fits")
    means = np.asarray([fit.mean for fit in fits])
    log_hbars = np.log(np.maximum([fit.hbar for fit in fits], 1e-4))
    alphas = np.asarray([fit.alpha for fit in fits])
    persistence = np.asarray([fit.persistence for fit in fits])
    alpha_low = float(np.clip(np.quantile(alphas, 0.10), 0.01, 0.40))
    alpha_high = float(np.clip(np.quantile(alphas, 0.90), alpha_low + 0.01, 0.40))
    persistence_low = float(
        np.clip(np.quantile(persistence, 0.10), alpha_high + 0.01, 0.985)
    )
    persistence_high = float(
        np.clip(np.quantile(persistence, 0.90), persistence_low + 0.005, 0.995)
    )
    return RegimePrior(
        mu_mean=float(np.median(means)),
        mu_sd=float(max(np.std(means, ddof=1), mu_sd_floor)),
        log_hbar_mean=float(np.mean(log_hbars)),
        log_hbar_sd=float(max(np.std(log_hbars, ddof=1), log_hbar_sd_floor)),
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        persistence_low=persistence_low,
        persistence_high=persistence_high,
    )
