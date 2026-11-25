# Local experiments

## Cube pick

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene pick_avbd_cube --solver avbd --viewer gl \
  --robot-collision-geometry finger-box-to-mesh \
  --rigid-soft-contact-use-log-barrier
```

## Shirt pick

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene shirt_pick --solver avbd --viewer gl \
  --robot-collision-geometry finger-box-to-mesh \
  --rigid-soft-contact-use-log-barrier
```

## Cloth punch

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene grasp_avbd_cloth --solver avbd --viewer gl --sequence quick_punch \
  --robot-collision-geometry finger-box-to-mesh \
  --rigid-soft-contact-use-log-barrier
```

## Mixed rigid and soft objects falling onto cloth

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene multiphysics_cloth_drop --solver avbd --viewer gl
```

## Cloth twist — penalty-DAT

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene cloth_twist --solver avbd --viewer gl --vbd-iterations 8
```

## Cloth catching a deformable ball — penalty-DAT

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene cloth_catch_ball --solver avbd --viewer gl
```

## Soft cubes on a rigid slope

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene soft_cube_slope --solver avbd --viewer gl
```

## Soft cube on a pinned soft slope — penalty-DAT

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene soft_cube_pinned_slope --solver avbd --viewer gl \
  --slope-mu 0.2 --slope-initial-speed 0.0
```

## Soft cube sliding to rest — penalty-DAT

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene soft_cube_flat_slide --solver avbd --viewer gl
```

## Sliding-to-rest ALM-DAT comparison

```bash
DISPLAY=:1 .venv/bin/python -m newton.exp \
  --scene soft_cube_flat_slide --solver avbd --viewer gl \
  --self-contact-method alm_dat
```

## Edge–box contact-force viewer

```bash
DISPLAY=:1 uv run --no-sync --with polyscope==2.6.1 python -m newton.exp.ogc_force_field \
  --probe edge --shape box
```
