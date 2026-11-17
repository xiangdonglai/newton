"""Publish the recorded example at the existing local illustration URL."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "illustrative-real.json").read_text())
    records = [r for r in data["checkpoints"] if r["step"] == 0]
    summary = data["summary"]
    assert [r["phase"] for r in records] == [
        "initialization",
        "iteration_1",
        "iteration_2",
        "final_round_1",
        "final_round_2",
    ]
    reference = records[0]["reference"]
    previous = reference
    for record in records:
        assert record["reference"] == reference
        assert record["anchor_before"] == previous
        assert record["vt_count"] == 1 and record["vt_valid"] == [1]
        assert record["vt_normal_before"] == [[0.0, 0.0, 1.0]]
        proposed = [
            a + record["truncation_t"] * (y - a) for a, y in zip(record["anchor_before"], record["trial"], strict=True)
        ]
        assert math.dist(proposed, record["anchor_after"]) < 1e-8
        assert math.dist(record["anchor_after"], reference) < summary["motion_budget_m"]
        boundary = record["vt_point_before"][0][2] + record["vt_margin_before"][0]
        assert record["anchor_after"][2] >= boundary
        previous = record["anchor_after"]
    assert records[-1]["trial"] == records[-2]["trial"] == records[-3]["trial"]
    assert records[-1]["normal_multipliers"] == records[-2]["normal_multipliers"] == records[-3]["normal_multipliers"]
    assert records[-1]["anchor_after"] == data["frames"][0]["position"]
    control = json.loads((ROOT / "illustrative-ten-iterations.json").read_text())
    for r in control["checkpoints"][-2:]:
        assert r["anchor_before"] == r["anchor_after"] == r["trial"]
    trace = {
        "summary": summary,
        "checkpoints": records,
        "frame": data["frames"][0],
        "control": {"iterations": 10, "frame": control["frames"][0]},
        "source_sha256": hashlib.sha256((ROOT / "run_example.py").read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256((ROOT / "illustrative-real.json").read_bytes()).hexdigest(),
    }
    template = (ROOT / "website-template.html").read_text()
    assert template.count("__RECORDED_TRACE__") == 1
    page = template.replace("__RECORDED_TRACE__", json.dumps(trace).replace("</", "<\\/"))
    (ROOT / "alm-dat-truncation.html").write_text(page)
    (ROOT / "alm-dat-selected-trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    verification = {
        "plane_limits_every_advance": True,
        "isotropic_budget_nonbinding": True,
        "ten_iteration_final_rounds_noop": True,
        "selected_step": 0,
        "selected_output_m": trace["frame"]["position"][2],
        "page_bytes": len(page.encode()),
    }
    (ROOT / "website-data-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
