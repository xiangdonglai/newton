Add opt-in dense rigid-soft mesh BVH queries to `CollisionPipeline`, with VT/TV/EE primitive identities, automatic soft-BVH refitting, and VBD force filtering for incorrectly oriented pairs.

Select the mesh backend with `rigid_soft_full_surface_mesh_backend`. With full-surface contact enabled, the default `"sdf"` backend now uses texture SDFs for mesh particle contacts as well as edge/face contacts. Particle-only contact retains its existing nearest-triangle queries.
