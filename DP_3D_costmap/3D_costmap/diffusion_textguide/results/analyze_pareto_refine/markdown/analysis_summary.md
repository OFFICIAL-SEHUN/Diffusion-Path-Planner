# Pareto sweep analysis summary

## Data loaded
- Raw JSON rows: **20580**; **20580** rows used after dropping non-finite `cot_mean` / `risk_mean` / `isr_mean`.
- Unique intents: **14**.
- Unique weight tuples `(α,β,γ,δ)`: **1470**.

## Pareto front over weight tuples `(α, β, γ, δ)`

- Non-dominated weight tuples (mean cot↓, mean risk↓, mean isr↑ over intents): **202**.
- See `tables/weight_pareto_front.csv` and `figures/3d/pareto_3d_weight_frontier.png`.

## Per-intent notes (same weights, fixed intent; use `--per-intent-pareto-plots` for figures)

### avoid_steep
- Pareto front size: **180**.
- CoT–ISR trade-off (on Pareto front): possibly weak (|ρ|≈0.02); objectives may be loosely coupled on this front.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.93); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.2, 2.2, 0.6, 2)`.

### baseline
- Pareto front size: **156**.
- CoT–ISR trade-off (on Pareto front): unclear (correlation undefined).
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.92); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.4, 1.5, 0.6, 2)`.

### center_bias
- Pareto front size: **156**.
- CoT–ISR trade-off (on Pareto front): unclear (correlation undefined).
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.92); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.4, 1.5, 0.6, 2)`.

### center_bias+prefer_flat
- Pareto front size: **16**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.29); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.86); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.2, 2.2, 0.6, 2)`.

### energy_efficient
- Pareto front size: **11**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈1.00); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.87); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.2, 2.2, 0.6, 2)`.

### energy_efficient+minimize_elevation_change
- Pareto front size: **48**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.86); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.78); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.2, 2.2, 0.4, 2)`.

### left_bias
- Pareto front size: **264**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.93); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): possibly weak (|ρ|≈0.01); objectives may be loosely coupled on this front.
- Heuristic balanced tuple (within intent Pareto front): `(0.3, 1.2, 0.6, 2)`.

### left_bias+avoid_steep
- Pareto front size: **195**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.95); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): possibly weak (|ρ|≈0.22); objectives may be loosely coupled on this front.
- Heuristic balanced tuple (within intent Pareto front): `(0.4, 1.4, 0.6, 2)`.

### minimize_elevation_change
- Pareto front size: **72**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.37); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.86); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.3, 2.2, 0.6, 2)`.

### prefer_flat
- Pareto front size: **455**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.96); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.59); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(1, 1.8, 0.5, 2)`.

### right_bias
- Pareto front size: **169**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.98); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.45); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.2, 2, 0.5, 2)`.

### right_bias+prefer_flat
- Pareto front size: **132**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.82); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.76); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.2, 2.2, 0.6, 2)`.

### short_path
- Pareto front size: **1080**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.66); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.88); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.6, 1.6, 0.3, 2)`.

### short_path+avoid_steep
- Pareto front size: **1080**.
- CoT–ISR trade-off (on Pareto front): notably structured (|ρ|≈0.70); interpret with terrain and sampling variability in mind.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.88); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.6, 1.6, 0.3, 2)`.

## Cross-intent summary

- Top row after sorting (high mean ISR, then low mean CoT / risk): `(0.2, 1.2, 0.2, 2.5)`.
- Rough correlation across grouped tuples between mean CoT and mean ISR: ρ ≈ **0.98** (cautious: not causal).
- A single `(α,β,γ,δ)` that is near-optimal for every intent is **unlikely**; see `cross_intent_tuple_selection` for `min_isr` / `min_feasibility` spreads.
- Intent-dependent geometry and penalties typically make trade-offs **context-specific** rather than universal.
- Instruction scores (ISR) and energy proxies (CoT) **can** move in opposite directions when grouped; verify on your sweep rather than assuming conflict.

## Recommended tuples (cross-intent)
- **Balanced:** `(0.2, 2, 0.5, 2.3)` (mean ISR=0.863, mean CoT=24.705).
- **Energy-favoring:** `(1, 2.2, 0.5, 2)` (mean ISR=0.830, mean CoT=23.200).
- **Instruction-favoring:** `(0.2, 1.2, 0.2, 2.5)` (mean ISR=0.878, mean CoT=26.196).
