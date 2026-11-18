Fix spurious ALM-DAT motion truncation from tiny approach rates and rounded
division planes. Choose interior planes when the full trial fits, validate
stored float32 clearance, and report uncertified pairs through
`SolverVBD.particle_dat_coverage_ok()`. Print warnings when separating-normal
search or stored-plane placement fails, including whether a previous plane
was retained.
