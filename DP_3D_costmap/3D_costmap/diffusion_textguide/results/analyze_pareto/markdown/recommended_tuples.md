# Recommended weight tuples

## Balanced (multi-gate utopia distance)

- **Weights:** `(0.5, 1.5, 0.5, 2.3)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 24.6698
- **mean_risk:** 8.9277
- **mean_isr:** 0.8582
- **min_isr:** 0.5784
- **Interpretation:** Balances mean CoT, mean risk, and mean ISR (after crossing gates on feasibility/ISR floors).

## Energy-favoring

- **Weights:** `(2.4, 1.8, 0.1, 2)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 22.2984
- **mean_risk:** 22.6316
- **mean_isr:** 0.8149
- **min_isr:** 0.5194
- **Interpretation:** Favors low mean CoT while keeping mean feasibility and ISR moderately high.

## Instruction-favoring

- **Weights:** `(0.5, 0.4, 0.1, 2.4)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 26.8189
- **mean_risk:** 18.1514
- **mean_isr:** 0.8809
- **min_isr:** 0.5495
- **Interpretation:** Favors high mean ISR with a feasibility floor; may trade higher CoT.
