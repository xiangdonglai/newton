"""Check the local recorded-state website in an isolated headless browser."""

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8767/alm-dat-truncation.html"


def main():
    trace = json.loads((ROOT / "alm-dat-selected-trace.json").read_text())
    records = trace["checkpoints"]
    state = [records[0]["reference"][2]] * 3 + [0.0]
    expected = [state.copy()]
    state[2] = records[0]["trial"][2]
    expected.append(state.copy())
    state[1] = records[0]["anchor_after"][2]
    expected.append(state.copy())
    for r in records[1:3]:
        state[2] = r["trial"][2]
        expected.append(state.copy())
        state[3] = r["normal_multipliers"][0]
        expected.append(state.copy())
        state[1] = r["anchor_after"][2]
        expected.append(state.copy())
    for r in records[3:]:
        state[1] = r["anchor_after"][2]
        expected.append(state.copy())
    expected.append(state.copy())
    errors = []
    requests = []
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/usr/bin/google-chrome", headless=True, args=["--disable-gpu", "--no-sandbox"]
        )
        page = browser.new_page(viewport={"width": 1260, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url))
        # Drive animation's display clock without waiting a minute for the walkthrough.
        page.add_init_script("""
            window.__frames = [];
            window.__time = 0;
            window.requestAnimationFrame = callback => { window.__frames.push(callback); return 1; };
            window.__advance = ms => {
                for (let t = 0; t < ms; t += 100) {
                    window.__time += 100;
                    const pending = window.__frames.splice(0);
                    for (const callback of pending) callback(window.__time);
                }
            };
        """)
        response = page.goto(URL)
        assert response.status == 200
        assert page.locator("#recorded-trace").text_content()
        assert len(expected) == 12
        for i, e in enumerate(expected):
            assert page.locator("#alm-dat-illustration").get_attribute("data-stage") == str(i)
            for key, value in zip(["r", "a", "y"], e[:3], strict=True):
                assert float(page.locator(f"#value-{key}").get_attribute("data-meters")) == value
            assert page.locator("#value-lambda").text_content() == f"{e[3]:.6f}"
            if i == 10:
                page.screenshot(path=str(ROOT / "website-final-round.png"), full_page=True)
            if i < len(expected) - 1:
                page.locator("#next-stage").click()
        assert page.locator("#next-stage").is_disabled()
        page.locator("#previous-stage").click()
        assert page.locator("#alm-dat-illustration").get_attribute("data-stage") == "10"
        for theme in ["light", "dark"]:
            page.emulate_media(color_scheme=theme)
            for width in [1260, 950, 760, 600, 390, 320]:
                page.set_viewport_size({"width": width, "height": 1100})
                page.wait_for_timeout(80)
                result = page.evaluate("""() => {
                    const issues = [];
                    for (const svg of document.querySelectorAll('.plot')) {
                        const bounds = svg.getBoundingClientRect();
                        for (const text of svg.querySelectorAll('text')) {
                            const box = text.getBoundingClientRect();
                            if (box.left < bounds.left - 1 || box.right > bounds.right + 1 ||
                                box.top < bounds.top - 1 || box.bottom > bounds.bottom + 1)
                                issues.push({text: text.textContent, kind: 'clipped SVG label'});
                            if (parseFloat(getComputedStyle(text).fontSize) < 11)
                                issues.push({text: text.textContent, kind: 'small text'});
                        }
                    }
                    for (const value of document.querySelectorAll('.state-values dd')) {
                        if (value.scrollWidth > value.clientWidth)
                            issues.push({text: value.textContent, kind: 'readout overflow'});
                    }
                    return {issues, overflow: document.documentElement.scrollWidth > window.innerWidth};
                }""")
                checks.append({"width": width, "theme": theme, **result})
                if width in [1260, 390]:
                    page.screenshot(path=str(ROOT / f"website-{theme}-{width}.png"), full_page=True)
                assert not result["overflow"], checks[-1]
                assert not result["issues"], checks[-1]
        # The deepest trial has the longest numeric label; check it at phone width too.
        for _ in range(9):
            page.locator("#previous-stage").click()
        assert page.locator("#alm-dat-illustration").get_attribute("data-stage") == "1"
        assert page.evaluate("""() => [...document.querySelectorAll('.state-values dd')]
            .every(value => value.scrollWidth <= value.clientWidth)""")
        # Check the entire automatic sequence and its final hold.
        page.reload()
        page.evaluate("ms => window.__advance(ms)", 60000)
        assert page.locator("#alm-dat-illustration").get_attribute("data-stage") == "11"
        assert "complete" in page.locator("#playback-status").text_content()
        page.emulate_media(reduced_motion="reduce")
        page.reload()
        assert page.locator("#alm-dat-illustration").get_attribute("data-stage") == "11"
        assert "Reduced-motion" in page.locator("#playback-status").text_content()
        # Run real requestAnimationFrame too, in a separate private page.
        live = browser.new_page(viewport={"width": 1260, "height": 1000})
        live.on("pageerror", lambda error: errors.append(str(error)))
        live.goto(URL)
        live.wait_for_function("document.getElementById('alm-dat-illustration').dataset.stage === '1'", timeout=8000)
        assert not errors, errors
        assert all(url.startswith("http://127.0.0.1:8767/") for url in requests), requests
        browser.close()
    result = {
        "stages_matched": len(expected),
        "layouts": checks,
        "page_errors": errors,
        "external_requests": [],
        "automatic_playback": True,
        "reduced_motion": True,
        "navigation": True,
        "real_animation_clock": True,
    }
    (ROOT / "website-browser-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
