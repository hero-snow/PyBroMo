# Journal of critical learnings - Bolt Persona

## ALEX Simulation
- To run an ALEX simulation correctly, `ParticlesSimulation` must be initialized with `t_step, t_max, particles, box, psf`.
- `simulate_diffusion` must be called with `total_emission=False` so that per-particle emission rates are available for `simulate_timestamps_alex`.
- `simulate_timestamps_alex` generates donor and acceptor timestamps based on ALEX period and duty cycles.

## Jupyter Notebooks
- Notebooks are JSON files with a specific structure (`cells`, `metadata`, `nbformat`).
- When creating notebooks programmatically, ensure `kernelspec` and `language_info` match the target environment.
