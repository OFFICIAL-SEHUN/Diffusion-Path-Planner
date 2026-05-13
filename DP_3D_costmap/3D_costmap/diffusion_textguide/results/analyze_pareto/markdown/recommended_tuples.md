# Recommended weight tuples

## Balanced (multi-gate utopia distance)

- **Weights:** `(2, 2, 2, 6)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 24.1616
- **mean_risk:** 19.1712
- **mean_isr:** 0.8922
- **min_isr:** 0.7139
- **Interpretation:** Balances mean CoT, mean risk, and mean ISR (after crossing gates on feasibility/ISR floors).

## Energy-favoring

- **Weights:** `(2, 2, 0, 2)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 21.9118
- **mean_risk:** 50.1408
- **mean_isr:** 0.7955
- **min_isr:** 0.5902
- **Interpretation:** Favors low mean CoT while keeping mean feasibility and ISR moderately high.

## Instruction-favoring

- **Weights:** `(0.5, 0, 0, 8)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 27.8056
- **mean_risk:** 50.7177
- **mean_isr:** 0.9375
- **min_isr:** 0.8399
- **Interpretation:** Favors high mean ISR with a feasibility floor; may trade higher CoT.
