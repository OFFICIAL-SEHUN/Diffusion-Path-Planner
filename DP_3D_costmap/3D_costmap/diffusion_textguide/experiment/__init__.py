"""
Experiment suite for Diffusion-based Text-conditioned Path Planner.

Modules:
  metrics          – Goal error, ISR, path length, CoT, risk, L2/Chamfer/Fréchet, cost gap
  utils            – Path ↔ pixel conversion, terrain loading, visualization helpers
  eval_pareto      – Intent cost weight Pareto frontier sweep  (Exp 1)
  eval_ablation    – Inner module ablation                      (Exp 2)
  eval_baselines   – External comparison with A* variants       (Exp 3)
  eval_similarity  – Pseudo-label similarity analysis           (Exp 4)
"""
