# Inference Time by Map Size (A* vs Diffusion)

| Map Size | A* (ms) | Diffusion (ms) | A*/Diff |
|----------|---------|----------------|--------|
| 32 | 42.13 | 1240.63 | 0.03x |
| 64 | 221.53 | 640.03 | 0.35x |
| 128 | 1838.34 | 596.46 | 3.08x |
| 256 | 15343.95 | 594.09 | 25.83x |

- **A***: Cost-aware A* planner.
- **Diffusion**: Neural diffusion-based planner (GPU/CPU).
- **A*/Diff**: Speed ratio (A* time / Diffusion time).
