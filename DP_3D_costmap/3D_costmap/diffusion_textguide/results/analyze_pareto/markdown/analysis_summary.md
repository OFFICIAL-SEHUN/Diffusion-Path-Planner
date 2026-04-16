# Pareto sweep analysis summary

## Data loaded
- Raw JSON rows: **8400**; **8400** rows used after dropping non-finite `cot_mean` / `risk_mean` / `isr_mean`.
- Unique intents: **14**.
- Unique weight tuples `(α,β,γ,δ)`: **600**.

## Pareto front over weight tuples `(α, β, γ, δ)`

- Non-dominated weight tuples (mean cot↓, mean risk↓, mean isr↑ over intents): **243**.
- See `tables/weight_pareto_front.csv` and `figures/3d/pareto_3d_weight_frontier.png`.

## Per-intent notes (same weights, fixed intent; use `--per-intent-pareto-plots` for figures)

### avoid_steep
- Pareto front size: **100**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.75); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.82); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 1.7, 0.2, 2)`.

### baseline
- Pareto front size: **100**.
- CoT–ISR trade-off (on Pareto front): unclear (correlation undefined).
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.80); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 1.7, 0.2, 2)`.

### center_bias
- Pareto front size: **100**.
- CoT–ISR trade-off (on Pareto front): unclear (correlation undefined).
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.80); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 1.7, 0.2, 2)`.

### center_bias+prefer_flat
- Pareto front size: **29**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.70); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.75); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(2.4, 1.7, 0.5, 2)`.

### energy_efficient
- Pareto front size: **27**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈1.00); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.65); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 1.8, 0.2, 2)`.

### energy_efficient+minimize_elevation_change
- Pareto front size: **62**.
- CoT–ISR trade-off (on Pareto front): possibly weak (|ρ|≈0.08); objectives may be loosely coupled on this front.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.79); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 1.8, 0.3, 2)`.

### left_bias
- Pareto front size: **191**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.97); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.27); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(2, 0.8, 0.4, 2.2)`.

### left_bias+avoid_steep
- Pareto front size: **130**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.96); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.51); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(1, 0.8, 0.3, 2.2)`.

### minimize_elevation_change
- Pareto front size: **62**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.48); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.65); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 1.8, 0.4, 2.1)`.

### prefer_flat
- Pareto front size: **266**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.99); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.94); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(1, 1.8, 0.5, 2)`.

### right_bias
- Pareto front size: **197**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.99); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.26); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(1, 0.8, 0.4, 2.1)`.

### right_bias+prefer_flat
- Pareto front size: **64**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.92); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.63); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(1, 0.4, 0.5, 2.1)`.

### short_path
- Pareto front size: **550**.
- CoT–ISR trade-off (on Pareto front): possibly weak (|ρ|≈0.11); objectives may be loosely coupled on this front.
- CoT–Risk trade-off (on Pareto front): possibly weak (|ρ|≈0.06); objectives may be loosely coupled on this front.
- Heuristic balanced tuple (within intent Pareto front): `(2.4, 1.5, 0.3, 2)`.

### short_path+avoid_steep
- Pareto front size: **520**.
- CoT–ISR trade-off (on Pareto front): possibly weak (|ρ|≈0.22); objectives may be loosely coupled on this front.
- CoT–Risk trade-off (on Pareto front): possibly weak (|ρ|≈0.12); objectives may be loosely coupled on this front.
- Heuristic balanced tuple (within intent Pareto front): `(2, 1.7, 0.3, 2)`.

## Cross-intent summary

- Top row after sorting (high mean ISR, then low mean CoT / risk): `(0.5, 0.4, 0.1, 2.4)`.
- Rough correlation across grouped tuples between mean CoT and mean ISR: ρ ≈ **0.97** (cautious: not causal).
- A single `(α,β,γ,δ)` that is near-optimal for every intent is **unlikely**; see `cross_intent_tuple_selection` for `min_isr` / `min_feasibility` spreads.
- Intent-dependent geometry and penalties typically make trade-offs **context-specific** rather than universal.
- Instruction scores (ISR) and energy proxies (CoT) **can** move in opposite directions when grouped; verify on your sweep rather than assuming conflict.

## Recommended tuples (cross-intent)
- **Balanced:** `(0.5, 1.5, 0.5, 2.3)` (mean ISR=0.858, mean CoT=24.670).
- **Energy-favoring:** `(2.4, 1.8, 0.1, 2)` (mean ISR=0.815, mean CoT=22.298).
- **Instruction-favoring:** `(0.5, 0.4, 0.1, 2.4)` (mean ISR=0.881, mean CoT=26.819).
