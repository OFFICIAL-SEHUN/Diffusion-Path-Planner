# Recommended weight tuples

## Balanced (multi-gate utopia distance)

- **Weights:** `(0.5, 1.5, 1.5, 1.5)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 24.3051
- **mean_risk:** 10.1017
- **mean_isr:** 0.8765
- **min_isr:** 0.7114
- **Interpretation:** Balances mean CoT, mean risk, and mean ISR (after crossing gates on feasibility/ISR floors).

## Energy-favoring

- **Weights:** `(1, 2.5, 0.5, 1.5)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 22.7187
- **mean_risk:** 21.3126
- **mean_isr:** 0.8291
- **min_isr:** 0.5703
- **Interpretation:** Favors low mean CoT while keeping mean feasibility and ISR moderately high.

## Instruction-favoring

- **Weights:** `(0.5, 1.5, 0.5, 2.5)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 25.6770
- **mean_risk:** 21.5525
- **mean_isr:** 0.8980
- **min_isr:** 0.7413
- **Interpretation:** Favors high mean ISR with a feasibility floor; may trade higher CoT.
