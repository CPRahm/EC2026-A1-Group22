"""Evaluate supplied weights on the John Set gecko in SimpleFlatWorld.

Run this file to check A-B-A repeatability. No evolution happens here.
MuJoCo's control callback is process-global: call evaluate sequentially;
future parallel experiments must use separate processes, not threads.
"""

from dataclasses import dataclass, field
from time import perf_counter, sleep

import mujoco as mj
import numpy as np
from numpy.typing import ArrayLike

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.simulation.environments import SimpleFlatWorld

# Support both VS Code's Run Python File and importing from the repository root.
if __package__:
    from .controller import ControllerSettings, NeuralController, sample_weights
else:
    from controller import ControllerSettings, NeuralController, sample_weights


@dataclass(frozen=True)
class EvaluationSettings:
    """Pilot settings, held constant across the methods being compared."""

    controller: ControllerSettings = field(default_factory=ControllerSettings)
    spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.1)
    target_position: tuple[float, float, float] = (2.0, 0.0, 0.1)
    duration: float = 15.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be finite and positive.")
        for position in (self.spawn_position, self.target_position):
            array = np.asarray(position, dtype=float)
            if array.shape != (3,) or not np.isfinite(array).all():
                raise ValueError("Positions must have three finite coordinates.")


@dataclass(frozen=True)
class EvaluationResult:
    fitness: float
    initial_position: tuple[float, float, float]
    final_position: tuple[float, float, float]
    simulated_seconds: float
    elapsed_seconds: float


class ViewerClosed(RuntimeError):
    """The preview was closed before its full evaluation finished."""


def _run_with_viewer(
    model: mj.MjModel,
    data: mj.MjData,
    steps: int,
    target: np.ndarray,
) -> None:
    """Step the measured simulation while the standard passive viewer displays it."""
    from mujoco import viewer

    # Keep this path deliberately close to MuJoCo's documented passive-viewer
    # example. The viewer does not advance physics; this function does.
    with viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as window:
        # Refresh approximately 30 times per simulated second while the
        # controller still runs every physics step, as in headless evaluation.
        frame_steps = max(1, round(1 / (30 * model.opt.timestep)))
        completed = 0
        wall_start = perf_counter()
        simulation_start = float(data.time)
        while completed < steps:
            if not window.is_running():
                raise ViewerClosed("Preview closed early; no visual-run score was returned.")
            batch = min(frame_steps, steps - completed)
            mj.mj_step(model, data, nstep=batch)
            completed += batch
            window.sync()
            # Slow the display to approximately real time, without changing dt.
            remaining = float(data.time) - simulation_start - (perf_counter() - wall_start)
            if remaining > 0:
                sleep(remaining)


def evaluate(
    weights: ArrayLike,
    settings: EvaluationSettings = EvaluationSettings(),
    *,
    show_viewer: bool = False,
) -> EvaluationResult:
    """Run a clean simulation and return final planar distance (lower is better).

    No weights are sampled here. Invalid inputs or simulation failures raise
    an exception, rather than silently entering the experimental results.
    elapsed_seconds includes controller setup, world compilation and simulation.
    With show_viewer=True it also includes window setup and display pacing;
    use the default without a viewer for runtime benchmarks and EA evaluations.
    """
    started = perf_counter()
    controller = NeuralController(weights, settings.controller)
    mj.set_mjcb_control(None)
    try:
        world = SimpleFlatWorld()
        robot = gecko()
        world.spawn(
            robot.spec,
            position=list(settings.spawn_position),
            correct_collision_with_floor=True,
        )
        model = world.spec.compile()
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

        # Fail explicitly if the chosen body no longer matches the network.
        if (model.nq, model.nu) != (
            settings.controller.qpos_size,
            settings.controller.output_size,
        ):
            raise ValueError("Compiled robot dimensions do not match controller settings.")
        if not np.allclose(model.actuator_ctrlrange, [-np.pi / 2, np.pi / 2]):
            raise ValueError("Robot actuator ranges do not match this controller.")

        requested_steps = settings.duration / model.opt.timestep
        steps = round(requested_steps)
        if steps < 1 or not np.isclose(requested_steps, steps, rtol=0, atol=1e-8):
            raise ValueError("duration must be a positive whole number of physics steps.")
        initial_position = tuple(float(x) for x in data.qpos[:3])

        target = np.asarray(settings.target_position, dtype=np.float64)

        def apply_control(_model: mj.MjModel, state: mj.MjData) -> None:
            state.ctrl[:] = controller.get_actions(state.qpos, state.time, target)

        mj.set_mjcb_control(apply_control)
        # The callback still runs at each physics step. Count steps explicitly
        # so evaluations cannot overshoot their duration by a runner batch.
        if show_viewer:
            _run_with_viewer(model, data, steps, target)
        else:
            mj.mj_step(model, data, nstep=steps)

        if np.any(data.warning.number):
            raise RuntimeError(f"MuJoCo reported simulation warnings: {data.warning.number}.")
        if not np.isclose(data.time, settings.duration, rtol=0, atol=1e-8):
            raise RuntimeError("Simulation did not complete its requested duration.")
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("Simulation ended in a non-finite state.")

        final_position = tuple(float(x) for x in data.qpos[:3])
        fitness = float(np.linalg.norm(
            np.asarray(final_position[:2]) - np.asarray(settings.target_position[:2])
        ))
        if not np.isfinite(fitness):
            raise FloatingPointError("Fitness is not finite.")
        return EvaluationResult(
            fitness=fitness,
            initial_position=initial_position,
            final_position=final_position,
            simulated_seconds=float(data.time),
            elapsed_seconds=perf_counter() - started,
        )
    finally:
        # Also detach after an error, so a later run cannot use old weights.
        mj.set_mjcb_control(None)


def main() -> None:
    """Small integration check, not an experimental run or learning pilot."""
    settings = EvaluationSettings()
    rng = np.random.default_rng(42)
    weights_a = sample_weights(rng, settings.controller)
    weights_b = sample_weights(rng, settings.controller)
    print(f"Raw qpos size: {settings.controller.qpos_size}")
    print(f"Inputs: {settings.controller.input_size}")
    print(f"Hidden neurons: {settings.controller.hidden_size}")
    print(f"Outputs: {settings.controller.output_size}")
    print(f"Flat genotype length: {settings.controller.genome_length}")

    results = []
    for label, weights in (("A", weights_a), ("B", weights_b), ("A again", weights_a)):
        result = evaluate(weights, settings)
        results.append(result)
        print(
            f"{label}: fitness={result.fitness:.4f}, "
            f"simulated={result.simulated_seconds:.3f}s, "
            f"elapsed={result.elapsed_seconds:.3f}s"
        )

    assert np.isclose(results[0].fitness, results[2].fitness, rtol=0, atol=1e-10)
    assert np.allclose(results[0].final_position, results[2].final_position, rtol=0, atol=1e-10)
    print("Repeatability check passed: same weights, same score and final position.")
    print("This checks integration only; the controller still needs a learning pilot.")


if __name__ == "__main__":
    main()
