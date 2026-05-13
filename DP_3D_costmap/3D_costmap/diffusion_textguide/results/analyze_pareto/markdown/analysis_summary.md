# Pareto sweep analysis summary

## Data loaded
- Raw JSON rows: **1920**; **1920** rows used after dropping non-finite `cot_mean` / `risk_mean` / `isr_mean`.
- Unique intents: **10**.
- Unique weight tuples `(α,β,γ,δ)`: **192**.

## Pareto front over weight tuples `(α, β, γ, δ)`

- Non-dominated weight tuples (mean cot↓, mean risk↓, mean isr↑ over intents): **103**.
- See `tables/weight_pareto_front.csv` and `figures/3d/pareto_3d_weight_frontier.png`.

## Per-intent notes (same weights, fixed intent; use `--per-intent-pareto-plots` for figures)

### avoid_steep
- Pareto front size: **46**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.74); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.94); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 2, 0.5, 2)`.

### baseline
- Pareto front size: **40**.
- CoT–ISR trade-off (on Pareto front): unclear (correlation undefined).
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.94); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(1, 2, 0.5, 2)`.

### center_bias
- Pareto front size: **24**.
- CoT–ISR trade-off (on Pareto front): possibly weak (|ρ|≈0.10); objectives may be loosely coupled on this front.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.61); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(2, 2, 2, 2)`.

### center_bias+prefer_flat
- Pareto front size: **24**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.73); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.73); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(2, 2, 2, 2)`.

### energy_efficient
- Pareto front size: **84**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈1.00); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.87); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(2, 2, 0.5, 2)`.

### left_bias
- Pareto front size: **63**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.95); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.34); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(1, 2, 1, 2)`.

### left_bias+avoid_steep
- Pareto front size: **45**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.83); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): possibly weak (|ρ|≈0.07); objectives may be loosely coupled on this front.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 2, 1, 2)`.

### prefer_flat
- Pareto front size: **57**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.94); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.43); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 2, 2, 2)`.

### right_bias
- Pareto front size: **64**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.94); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.42); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(2, 0.5, 1, 4)`.

### right_bias+prefer_flat
- Pareto front size: **73**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.86); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.68); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 0, 2, 2)`.

## Cross-intent summary

- Top row after sorting (high mean ISR, then low mean CoT / risk): `(0.5, 0, 0, 8)`.
- Rough correlation across grouped tuples between mean CoT and mean ISR: ρ ≈ **0.92** (cautious: not causal).
- A single `(α,β,γ,δ)` that is near-optimal for every intent is **unlikely**; see `cross_intent_tuple_selection` for `min_isr` / `min_feasibility` spreads.
- Intent-dependent geometry and penalties typically make trade-offs **context-specific** rather than universal.
- Instruction scores (ISR) and energy proxies (CoT) **can** move in opposite directions when grouped; verify on your sweep rather than assuming conflict.

## Recommended tuples (cross-intent)
- **Balanced:** `(2, 2, 2, 6)` (mean ISR=0.892, mean CoT=24.162).
- **Energy-favoring:** `(2, 2, 0, 2)` (mean ISR=0.795, mean CoT=21.912).
- **Instruction-favoring:** `(0.5, 0, 0, 8)` (mean ISR=0.937, mean CoT=27.806).
