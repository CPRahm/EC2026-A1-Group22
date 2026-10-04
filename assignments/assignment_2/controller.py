"""A2 pilot controller: joint state, orientation, target offset and a clock.

This module is used by evaluation.py; running it directly starts no simulation.
The network is a feedforward neural network. Periodic inputs provide timing,
but no walking pattern is prescribed. All weights and biases can be evolved.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ControllerSettings:
    """Pilot choices; keep identical across EA conditions and random search."""

    joint_count: int = 6
    hidden_size: int = 6
    initial_weight_std: float = 0.5
    frequency_hz: float = 1.0
    target_distance_scale: float = 2.0

    def __post_init__(self) -> None:
        for size in (self.joint_count, self.hidden_size):
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError("Network sizes must be positive integers.")
        for name in ("initial_weight_std", "frequency_hz", "target_distance_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

    @property
    def qpos_size(self) -> int:
        """Free core position (3), orientation quaternion (4), then joints."""
        return 7 + self.joint_count

    @property
    def input_size(self) -> int:
        """Joint angles + orientation (4) + target offset (2) + clock (2)."""
        return self.joint_count + 8

    @property
    def output_size(self) -> int:
        return self.joint_count

    @property
    def genome_length(self) -> int:
        """W1, hidden biases, W2, output biases: 132 parameters by default."""
        return (
            self.input_size * self.hidden_size
            + self.hidden_size
            + self.hidden_size * self.output_size
            + self.output_size
        )


def sample_weights(
    rng: np.random.Generator,
    settings: ControllerSettings = ControllerSettings(),
) -> NDArray[np.float64]:
    """Sample all weights and biases; the caller owns and seeds rng."""
    return rng.normal(0.0, settings.initial_weight_std, size=settings.genome_length)


def decode_weights(
    weights: ArrayLike,
    settings: ControllerSettings = ControllerSettings(),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Flat layout: row-major W1, b1, row-major W2, b2. Copy caller data."""
    flat = np.array(weights, dtype=np.float64, copy=True)
    if flat.shape != (settings.genome_length,):
        raise ValueError(
            f"Expected a flat vector of {settings.genome_length} parameters; got {flat.shape}."
        )
    if not np.isfinite(flat).all():
        raise ValueError("All weights and biases must be finite.")
    end_w1 = settings.input_size * settings.hidden_size
    end_b1 = end_w1 + settings.hidden_size
    end_w2 = end_b1 + settings.hidden_size * settings.output_size
    w1 = flat[:end_w1].reshape(settings.input_size, settings.hidden_size)
    b1 = flat[end_w1:end_b1]
    w2 = flat[end_b1:end_w2].reshape(settings.hidden_size, settings.output_size)
    b2 = flat[end_w2:]
    return w1, b1, w2, b2


def make_inputs(
    qpos: ArrayLike,
    simulation_time: float,
    target_position: ArrayLike,
    settings: ControllerSettings = ControllerSettings(),
) -> NDArray[np.float64]:
    """Build the documented 14 inputs from MuJoCo's 13 qpos values and task data.

    Input order: joint angles / (pi/2); core quaternion (w,x,y,z);
    world-frame target offset (dx,dy) / target_distance_scale; sin and cos phase.
    The quaternion comes from MuJoCo's free joint and is already normalized.
    """
    state = np.asarray(qpos, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    if state.shape != (settings.qpos_size,):
        raise ValueError(f"Expected {settings.qpos_size} qpos values; got {state.shape}.")
    if target.shape != (3,):
        raise ValueError("target_position must have three coordinates.")
    if not np.isfinite(state).all() or not np.isfinite(target).all():
        raise FloatingPointError("State and target coordinates must be finite.")
    if not np.isfinite(simulation_time) or simulation_time < 0:
        raise ValueError("simulation_time must be finite and non-negative.")

    n = settings.joint_count
    inputs = np.empty(settings.input_size, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        inputs[:n] = state[7:] / (np.pi / 2)
        inputs[n:n + 4] = state[3:7]
        inputs[n + 4:n + 6] = (target[:2] - state[:2]) / settings.target_distance_scale
        phase = 2 * np.pi * settings.frequency_hz * simulation_time
        inputs[-2:] = np.sin(phase), np.cos(phase)
    if not np.isfinite(inputs).all():
        raise FloatingPointError("Controller inputs are not finite.")
    return inputs


class NeuralController:
    """Use one candidate's fixed parameters throughout one simulation."""

    def __init__(
        self,
        weights: ArrayLike,
        settings: ControllerSettings = ControllerSettings(),
    ) -> None:
        self.settings = settings
        self.w1, self.b1, self.w2, self.b2 = decode_weights(weights, settings)

    def get_actions(
        self,
        qpos: ArrayLike,
        simulation_time: float,
        target_position: ArrayLike,
    ) -> NDArray[np.float64]:
        """Return six direct hinge-angle commands in [-pi/2, pi/2]."""
        inputs = make_inputs(qpos, simulation_time, target_position, self.settings)
        with np.errstate(over="raise", invalid="raise"):
            hidden = np.tanh(inputs @ self.w1 + self.b1)
            actions = np.tanh(hidden @ self.w2 + self.b2) * (np.pi / 2)
        if not np.isfinite(actions).all():
            raise FloatingPointError("Controller produced non-finite actions.")
        return actions
