"""Exercise the paused viewer's Step and Reset requests on a private display."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SITE = Path(__file__).resolve().parents[3] / "tmp" / "alm-dat"
spec = importlib.util.spec_from_file_location("recorded_example", SITE / "run_particle_triangle.py")
example_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example_module)
import newton.viewer  # noqa: E402

expected = json.loads((ROOT / "illustrative-real.json").read_text())
checkpoints = [r for r in expected["checkpoints"] if r["step"] == 0]
evidence = {}


class CheckedViewer(newton.viewer.ViewerGL):
    """Use the same requests as the GUI buttons, leaving stepping unchanged."""

    checks = 0

    def is_running(self):
        self.checks += 1
        example = self._reset_callback.__self__
        assert self.is_paused()
        assert not self.should_step()
        if self.checks == 1:
            assert len(example.frames) == 0 and example.sim_time == 0.0
            assert example.args.dt == 0.12 and example.args.iterations == 2
            assert example.args.height == 0.04 and example.args.contact_gap == 0.2
            assert example.args.steps == 120
            self._step_requested = True  # Exact request set by the native Step button.
            return super().is_running()
        assert example.frames == expected["frames"][:1]
        assert example.solver.records == checkpoints
        assert example.sim_time == 0.12
        if self.checks == 2:
            from PIL import Image

            evidence["first_step"] = {
                "time_s": example.sim_time,
                "height_m": example.frames[0]["position"][2],
                "checkpoints_matched": len(checkpoints),
            }
            Image.fromarray(self.get_frame().numpy()).save(ROOT / "viewer-website-first-step.png")
            self._reset_callback()  # Exact callback invoked by the native Reset button.
            assert not example.frames and not example.solver.records and example.sim_time == 0.0
            np.testing.assert_array_equal(example.state.particle_q.numpy(), example.model.particle_q.numpy())
            assert self.is_paused()
            self._step_requested = True
            return super().is_running()
        evidence["reset_and_repeated_first_step"] = "Exactly matches the website trace; pause preserved."
        return False


newton.viewer.ViewerGL = CheckedViewer
sys.argv = [
    str(SITE / "run_particle_triangle.py"),
    "--device",
    "cpu",  # This verifier compares against the original Warp 1.19.0 CPU recording.
    "--viewer",
    "gl",
    "--paused",
    "--output",
    str(ROOT / "result-website-viewer.json"),
]
example_module.main()
(ROOT / "viewer-website-config-verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(json.dumps(evidence, indent=2))
