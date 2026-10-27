# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the viewer's body probe with an isolated production EE formulation.

Newton compiles NEWTON_ESP_EE_QUADRATURE once per process. A persistent child
lets the viewer switch formulations without changing compiled Warp constants.
Only poses/settings and small results cross the pipe; geometry stays in the
child. This file defines no contact energy or derivatives.
"""

import atexit
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


class ProductionBodyProbe:
    def __init__(self, scene, device, quadrature):
        env = dict(os.environ, NEWTON_ESP_EE_QUADRATURE=quadrature)
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), str(scene), str(device)],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        atexit.register(self.close)

    def evaluate(self, transform_np, support, stiffness, com_np, ee_only=False):
        request = [np.asarray(transform_np).tolist(), support, stiffness, np.asarray(com_np).tolist(), bool(ee_only)]
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError(f"Production ESP evaluator exited ({self.process.poll()}); see terminal output")
        result = json.loads(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        energy, force, torque, info = result["result"]
        for name in ("energy_terms", "positions", "com", "esp_wrench", "native_wrench"):
            if name in info:
                info[name] = np.asarray(info[name])
        return energy, np.asarray(force), np.asarray(torque), info

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # This evaluator belongs to us, not another user's viewer/job.
                self.process.terminate()
                self.process.wait(timeout=5)
            self.process.stdout.close()
        atexit.unregister(self.close)


def main():
    # Reserve stdout for responses, including while Warp imports/compiles.
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from view_esp_force_field import BodyForceProbe  # noqa: PLC0415

        probe = BodyForceProbe(int(sys.argv[1]), sys.argv[2])
        for line in sys.stdin:
            try:
                pose, support, stiffness, com, ee_only = json.loads(line)
                probe.model.body_com.assign(np.asarray(com, dtype=np.float32))
                result = probe.evaluate_production(np.asarray(pose), support, stiffness, ee_only=ee_only)
                response = {"result": result}
            except Exception as error:
                response = {"error": f"{type(error).__name__}: {error}"}
            output.write(json.dumps(response, default=lambda x: x.tolist()) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
