"""Keras LSTM, Sharpe objectives, and diversified validation callback."""

from __future__ import annotations

import os
from collections.abc import Callable

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import ops
import numpy as np

from .config import SEQUENCE_LENGTH


def gross_sharpe_loss(y_true: object, y_pred: object) -> object:
    """Return the negative annualised gross Sharpe training objective.

    Args:
        y_true: Tensor containing target return and validity mask channels.
        y_pred: Tensor containing bounded predicted positions.

    Returns:
        Scalar differentiable loss tensor.
    """
    target, mask = y_true[..., 0], y_true[..., 1]
    captured = y_pred[..., 0] * target * mask
    count = ops.sum(mask) + 1e-9
    mean = ops.sum(captured) / count
    variance = ops.sum(mask * (captured - mean) ** 2) / count
    return -ops.sqrt(252.0) * mean / ops.sqrt(variance + 1e-9)


def net_sharpe_loss(cost_bps: float) -> Callable[[object, object], object]:
    """Create a negative net-Sharpe loss at a fixed transaction-cost rate.

    Args:
        cost_bps: Cost per unit of absolute holding change, in basis points.

    Returns:
        Keras-compatible loss callable.
    """
    rate = float(cost_bps) * 1e-4

    def loss(y_true: object, y_pred: object) -> object:
        """Evaluate net Sharpe for one target and prediction batch."""
        target, mask, leverage = y_true[..., 0], y_true[..., 1], y_true[..., 2]
        position = y_pred[..., 0]
        captured = position * target * mask
        holding = position * leverage * mask
        turnover = (
            ops.absolute(holding[:, 1:] - holding[:, :-1])
            * mask[:, 1:]
            * mask[:, :-1]
        )
        costs = ops.concatenate((ops.zeros_like(turnover[:, :1]), rate * turnover), axis=1)
        net = captured - costs
        count = ops.sum(mask) + 1e-9
        mean = ops.sum(net) / count
        variance = ops.sum(mask * (net - mean) ** 2) / count
        return -ops.sqrt(252.0) * mean / ops.sqrt(variance + 1e-9)

    return loss


def build_lstm(
    n_features: int,
    hyperparameters: dict[str, float],
    cost_bps: float = 0.0,
) -> keras.Model:
    """Construct and compile the dissertation LSTM position model.

    Args:
        n_features: Number of input features at each sequence step.
        hyperparameters: Mapping containing ``hidden``, ``dropout``, ``lr``,
            and ``clipnorm``.
        cost_bps: Transaction-cost rate embedded in the training objective.

    Returns:
        Compiled Keras sequence model with bounded scalar positions.
    """
    inputs = keras.Input((SEQUENCE_LENGTH, n_features))
    hidden = keras.layers.LSTM(
        int(hyperparameters["hidden"]),
        return_sequences=True,
        dropout=float(hyperparameters["dropout"]),
        recurrent_dropout=0.0,
    )(inputs)
    hidden = keras.layers.Dropout(float(hyperparameters["dropout"]))(hidden)
    outputs = keras.layers.Dense(
        1,
        activation="tanh",
        kernel_constraint=keras.constraints.MaxNorm(3.0),
    )(hidden)
    model = keras.Model(inputs, outputs)
    model.compile(
        loss=net_sharpe_loss(cost_bps) if cost_bps else gross_sharpe_loss,
        optimizer=keras.optimizers.Adam(
            learning_rate=float(hyperparameters["lr"]),
            clipnorm=float(hyperparameters["clipnorm"]),
        ),
    )
    return model


class DiversifiedValidationSharpe(keras.callbacks.Callback):
    """Select weights using the date-level diversified validation portfolio.

    Args:
        x_valid: Validation feature sequences.
        y_valid: Validation target, mask, and leverage channels.
        date_codes: Flattened integer date code for every sequence position.
        date_count: Number of distinct validation dates.
        patience: Epochs without improvement before stopping.
        cost_bps: Transaction-cost rate used in validation selection.
        min_delta: Minimum Sharpe improvement counted as progress.
    """

    def __init__(
        self,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        date_codes: np.ndarray,
        date_count: int,
        *,
        patience: int,
        cost_bps: float,
        min_delta: float = 1e-4,
    ) -> None:
        """Store validation arrays and initialise early-stopping state."""
        super().__init__()
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.date_codes = date_codes
        self.date_count = int(date_count)
        self.patience = int(patience)
        self.cost_bps = float(cost_bps)
        self.min_delta = float(min_delta)
        self.best = -np.inf
        self.best_epoch = -1
        self.best_weights = None
        self.wait = 0
        self.history: list[float] = []

    def validation_sharpe(self) -> float:
        """Calculate the current model's diversified validation Sharpe ratio."""
        positions = self.model.predict(self.x_valid, batch_size=512, verbose=0)[..., 0]
        target = self.y_valid[..., 0]
        mask = self.y_valid[..., 1]
        captured = positions * target * mask
        if self.cost_bps:
            holding = positions * self.y_valid[..., 2] * mask
            turnover = np.abs(np.diff(holding, axis=1)) * mask[:, 1:] * mask[:, :-1]
            captured[:, 1:] -= self.cost_bps * 1e-4 * turnover
        flat_captured, flat_mask = captured.ravel(), mask.ravel()
        numerator = np.bincount(
            self.date_codes, weights=flat_captured, minlength=self.date_count
        )
        denominator = np.bincount(
            self.date_codes, weights=flat_mask, minlength=self.date_count
        )
        daily = numerator[denominator > 0] / denominator[denominator > 0]
        if len(daily) < 2 or float(daily.std(ddof=0)) <= 0:
            return -np.inf
        return float(daily.mean() / daily.std(ddof=0) * np.sqrt(252.0))

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """Update best weights and stop after the configured non-improvement run."""
        if logs and not np.isfinite(logs.get("loss", np.nan)):
            self.model.stop_training = True
            return
        sharpe = self.validation_sharpe()
        self.history.append(sharpe)
        if sharpe > self.best + self.min_delta:
            self.best = sharpe
            self.best_epoch = int(epoch)
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        """Restore the weights with the highest validation Sharpe ratio."""
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
