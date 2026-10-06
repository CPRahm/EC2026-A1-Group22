"""Watch the current A2 controller using the same physics as the evaluator.

Run this file with VS Code's Run Python File button. A 15-second preview opens
after a reference evaluation. The target is evaluated numerically. The viewer
displays the measured simulation state directly and does not advance physics by itself.
This is a debugging helper, not the final saved-controller replay tool.
"""

import numpy as np

if __package__:
    from .controller import sample_weights
    from .evaluation import EvaluationSettings, ViewerClosed, evaluate
else:
    from controller import sample_weights
    from evaluation import EvaluationSettings, ViewerClosed, evaluate


def main() -> None:
    settings = EvaluationSettings()
    rng = np.random.default_rng(42)
    weights = sample_weights(rng, settings.controller)

    print("Checking one random controller, with seed 42.")
    reference = evaluate(weights, settings)
    print(f"Reference fitness without a window: {reference.fitness:.4f}")
    print("Opening the preview. The target is evaluated numerically.")
    print("Watch the joints, body and floor contact; purposeful walking is not expected yet.")
    print("The preview runs automatically and closes after 15 simulated seconds.")
    try:
        visual = evaluate(weights, settings, show_viewer=True)
    except ViewerClosed as error:
        print(error)
        print("Run this file again and let the preview finish to check matching results.")
        return

    print(f"Visual-run fitness: {visual.fitness:.4f}")
    print(f"Simulated duration: {visual.simulated_seconds:.3f}s")
    print(f"Final position: {np.round(visual.final_position, 4)}")
    assert np.isclose(reference.fitness, visual.fitness, rtol=0, atol=1e-10), (
        "The preview and headless fitness differed. Investigate before experiments."
    )
    assert np.allclose(reference.final_position, visual.final_position, rtol=0, atol=1e-10), (
        "The preview and headless final positions differed."
    )
    print("Visual/headless check passed: same score and final position.")
    print("This confirms matching simulation results; learning still needs a pilot.")


if __name__ == "__main__":
    main()
    