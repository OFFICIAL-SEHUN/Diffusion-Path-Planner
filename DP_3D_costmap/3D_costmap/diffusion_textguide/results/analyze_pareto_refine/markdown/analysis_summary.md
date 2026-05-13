# Pareto sweep analysis summary

## Data loaded
- Raw JSON rows: **6250**; **6250** rows used after dropping non-finite `cot_mean` / `risk_mean` / `isr_mean`.
- Unique intents: **10**.
- Unique weight tuples `(α,β,γ,δ)`: **625**.

## Pareto front over weight tuples `(α, β, γ, δ)`

- Non-dominated weight tuples (mean cot↓, mean risk↓, mean isr↑ over intents): **200**.
- See `tables/weight_pareto_front.csv` and `figures/3d/pareto_3d_weight_frontier.png`.

## Per-intent notes (same weights, fixed intent; use `--per-intent-pareto-plots` for figures)

### avoid_steep
- Pareto front size: **168**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.87); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.68); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.25, 2.25, 1.5, 1.5)`.

### baseline
- Pareto front size: **175**.
- CoT–ISR trade-off (on Pareto front): unclear (correlation undefined).
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.68); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.25, 2.25, 1.5, 1.5)`.

### center_bias
- Pareto front size: **52**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.91); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.66); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 1.75, 1.5, 1.5)`.

### center_bias+prefer_flat
- Pareto front size: **27**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.91); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.61); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.75, 2.5, 1.5, 1.5)`.

### energy_efficient
- Pareto front size: **75**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈1.00); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.81); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.75, 1.5, 1.5, 1.5)`.

### left_bias
- Pareto front size: **126**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.96); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.26); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.75, 1.5, 1.25, 1.75)`.

### left_bias+avoid_steep
- Pareto front size: **222**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.93); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.42); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 2, 1.5, 2)`.

### prefer_flat
- Pareto front size: **220**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.96); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.85); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.25, 2.5, 1.5, 1.5)`.

### right_bias
- Pareto front size: **135**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.62); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): possibly weak (|ρ|≈0.25); objectives may be loosely coupled on this front.
- Heuristic balanced tuple (within intent Pareto front): `(0.5, 2.5, 1.25, 1.5)`.

### right_bias+prefer_flat
- Pareto front size: **140**.
- CoT–ISR trade-off (on Pareto front): moderate (|ρ|≈0.60); spread across the front suggests non-trivial trade-offs.
- CoT–Risk trade-off (on Pareto front): moderate (|ρ|≈0.80); spread across the front suggests non-trivial trade-offs.
- Heuristic balanced tuple (within intent Pareto front): `(0.25, 2.5, 1.25, 1.5)`.

## Cross-intent summary

- Top row after sorting (high mean ISR, then low mean CoT / risk): `(0.5, 1.5, 0.5, 2.5)`.
- Rough correlation across grouped tuples between mean CoT and mean ISR: ρ ≈ **-0.47** (cautious: not causal).
- A single `(α,β,γ,δ)` that is near-optimal for every intent is **unlikely**; see `cross_intent_tuple_selection` for `min_isr` / `min_feasibility` spreads.
- Intent-dependent geometry and penalties typically make trade-offs **context-specific** rather than universal.
- Instruction scores (ISR) and energy proxies (CoT) **can** move in opposite directions when grouped; verify on your sweep rather than assuming conflict.

## Recommended tuples (cross-intent)
- **Balanced:** `(0.5, 1.5, 1.5, 1.5)` (mean ISR=0.876, mean CoT=24.305).
- **Energy-favoring:** `(1, 2.5, 0.5, 1.5)` (mean ISR=0.829, mean CoT=22.719).
- **Instruction-favoring:** `(0.5, 1.5, 0.5, 2.5)` (mean ISR=0.898, mean CoT=25.677).
