Fix spurious ALM-DAT motion truncation from tiny approach rates and rounded
division planes. Choose interior planes when the full trial fits, validate
stored float32 clearance, and report uncertified pairs through
`SolverVBD.particle_dat_coverage_ok()`. Print warnings when separating-normal
search or stored-plane placement fails, including whether a previous plane
was retained.
For thin, strictly separated self-contact pairs, cap the per-side margin at
half the projected anchor gap. Clip against the positive clearance that the
stored plane actually provides, up to that margin.
The internal plane-failure fallback can retry along a second normal and hold
uncovered endpoints. By default, a pair without a plane contributes no constraint.
