# ALM-DAT: one particle and a fixed triangle

This local example runs the branch's real soft self-contact ALM-DAT solver.
Three zero-mass vertices form a fixed triangle at height zero; a free particle
falls under gravity. The website shows the recorded first physical step,
including its two VBD iterations and both final DAT rounds.

## Newton viewer

From the repository root:

```bash
DISPLAY=:1 uv run --script tmp/alm-dat/run_particle_triangle.py \
  --viewer gl --speed 0.1 --paused
```

Defaults match the website: timestep 0.12 s, two VBD iterations, initial height
0.04 m, detection gap 0.2 m, and 120 physical steps. These deliberately coarse
settings make the difference between the trial and anchor easy to see.

Click **Step** once to reach time 0.12 s and particle center height 0.005037 mm.
**Reset** restores the initial positions, velocity, solver history, and time.
Space pauses/resumes; period advances one step while paused. The viewer displays
accepted positions after complete physical steps. The website displays the
internal checkpoints within the first step. The sphere has a 5 mm display radius;
its center height is the simulated quantity.

The particle later rebounds and settles about 19.019 mm above the triangle,
where the compliant ALM contact force balances gravity. This is different from
the first step's 5.037 micrometer accepted height.

To reproduce the earlier small-timestep example, append:

```bash
--dt 0.016666666666666666 --iterations 10 --height 0.06 --contact-gap 0.01 --steps 240
```

The script declares its Python dependencies for `uv`; it imports Newton from this
checkout. Runs use stable Warp 1.17.0 on `cuda:0` by default; pass `--device cpu`
to use the CPU. The panel and saved summary show the actual device. Closing the
viewer saves `result-website-config.json` beside the script. Override that path
with `--output`. The website retains its original Warp 1.19.0 CPU recording.

## Website

The checked-in page can be opened directly as
[alm-dat-truncation.html](../../../tmp/alm-dat/alm-dat-truncation.html), or served on the LAN:

```bash
uv run --no-project python -m http.server 8767 --bind 0.0.0.0 --directory tmp/alm-dat
```

Visit `http://<this-machine-IP>:8767/alm-dat-truncation.html`.
The existing server is available at
<http://10.110.51.230:8767/alm-dat-truncation.html>.
Do not start another server if port 8767 is already occupied.

Edit `website-template.html` and rebuild the page from the recorded data:

```bash
uv run --no-project python ctx/2026-11-17-minimal-alm-dat/website-support/build_website.py
```

The builder validates the five recorded DAT advances and the ten-iteration
comparison. The traces and template live here in untracked `ctx`; the builder reads the
tracked script from `tmp/alm-dat` and writes the rendered page there. It produces the standalone page in `tmp/alm-dat` and keeps the selected-substep
JSON with source and trace hashes here in `ctx`. The original [schematic](alm-dat-truncation-schematic.html)
and [mathematical comparison](comparison.md) are also included.

## Recorded data and verification

`illustrative-real.json` contains the 120-step trace recorded at solver commit
`cfd6e07b50cdaca7deecd17e4a36709da12424ec`. Its formatting is compact to keep the
file below 500 KiB; no values or checkpoints were removed.
`illustrative-ten-iterations.json` contains the one-step comparison with ten VBD
iterations. `alm-dat-selected-trace.json` contains the first step used by the page.

Run the current code without a viewer, preserving those recorded inputs:

```bash
uv run --script tmp/alm-dat/run_particle_triangle.py
```

`verify_website.py` checks the stage values, layout, navigation, and animation
against the recorded data. With the server running and Google Chrome installed
at `/usr/bin/google-chrome`:

```bash
uv run --no-project --with playwright python ctx/2026-11-17-minimal-alm-dat/website-support/verify_website.py
```

`verify_website_viewer.py` checks the native paused Step and Reset requests against
the recorded first step. To use a private display rather than the shared desktop:

```bash
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a uv run --no-project --python 3.12 \
  --with warp-lang==1.19.0 --with numpy --with 'pyglet>=2.1.6,<3' \
  --with 'imgui_bundle>=1.92.0' --with PyOpenGL --with Pillow \
  python ctx/2026-11-17-minimal-alm-dat/website-support/verify_website_viewer.py
```

These auxiliary sources, traces, reports, and screenshots are untracked.
Only the runnable script and rendered HTML page remain tracked in `tmp/alm-dat`.
