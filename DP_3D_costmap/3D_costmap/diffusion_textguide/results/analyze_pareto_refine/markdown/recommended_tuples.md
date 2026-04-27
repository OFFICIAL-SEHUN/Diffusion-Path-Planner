# Recommended weight tuples

## Balanced (multi-gate utopia distance)

- **Weights:** `(0.2, 2, 0.5, 2.3)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 24.7051
- **mean_risk:** 8.7948
- **mean_isr:** 0.8630
- **min_isr:** 0.5768
- **Interpretation:** Balances mean CoT, mean risk, and mean ISR (after crossing gates on feasibility/ISR floors).

## Energy-favoring

- **Weights:** `(1, 2.2, 0.5, 2)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 23.2001
- **mean_risk:** 8.5753
- **mean_isr:** 0.8302
- **min_isr:** 0.5438
- **Interpretation:** Favors low mean CoT while keeping mean feasibility and ISR moderately high.

## Instruction-favoring

- **Weights:** `(0.2, 1.2, 0.2, 2.5)`
- **mean_feasibility:** 1.0000
- **mean_cot:** 26.1958
- **mean_risk:** 12.5901
- **mean_isr:** 0.8780
- **min_isr:** 0.5780
- **Interpretation:** Favors high mean ISR with a feasibility floor; may trade higher CoT.
