"""
A* vs Diffusion inference time benchmark.
- Measures A* pathfinding time (CPU).
- Optionally measures Diffusion sampling time (GPU/CPU) for comparison.
"""
import argparse
import os
import random
import time
import yaml
import numpy as np

from maze import MazeGenerator, a_star_search


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    img_size = config["data"]["img_size"]
    scale = config["maze"]["scale"]
    cost_weight = config["data"]["cost_weight"]
    num_runs = args.num_runs

    # --- Generate costmaps once (same maps for A* and optionally for diffusion) ---
    print(f"Generating {num_runs} costmaps for benchmarking...")
    maze_gen = MazeGenerator(img_size, scale)
    samples = []
    for _ in range(num_runs):
        costmap, _, start_pos, end_pos = maze_gen.generate(cost_weight=cost_weight)
        if isinstance(costmap, np.ndarray):
            costmap = np.where(np.isinf(costmap), 1.0, costmap).astype(np.float32)
        samples.append((costmap, tuple(start_pos), tuple(end_pos)))
    print(f"Done. Benchmarking A* ({num_runs} runs)...")

    # --- A* inference time ---
    a_star_times = []
    for costmap, start_pos, end_pos in samples:
        start_pos = tuple(start_pos)
        end_pos = tuple(end_pos)
        t0 = time.perf_counter()
        path = a_star_search(costmap, start_pos, end_pos, cost_weight=cost_weight)
        t1 = time.perf_counter()
        a_star_times.append(t1 - t0)

    a_star_times = np.array(a_star_times) * 1000  # ms
    mean_astar = np.mean(a_star_times)
    std_astar = np.std(a_star_times)
    print(f"\n--- A* inference time ---")
    print(f"  Mean: {mean_astar:.4f} ms")
    print(f"  Std:  {std_astar:.4f} ms")
    print(f"  Min:  {np.min(a_star_times):.4f} ms  |  Max: {np.max(a_star_times):.4f} ms")

    # --- Optional: Diffusion inference time (single run or average over same costmaps) ---
    if args.diffusion:
        import torch
        from model import ConditionalPathModel
        from diffusion import DiffusionScheduler

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading Diffusion model (device: {device})...")
        model = ConditionalPathModel(config=config)
        model_path = os.path.join(
            config["training"]["checkpoint_dir"],
            config["training"]["model_name"],
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        infer_timesteps = config.get("inference", {}).get(
            "timesteps", config["diffusion"]["timesteps"]
        )
        diffusion_scheduler = DiffusionScheduler(
            timesteps=infer_timesteps,
            beta_start=config["diffusion"]["beta_start"],
            beta_end=config["diffusion"]["beta_end"],
            device=device,
        )

        num_diff_runs = min(args.num_runs, 5)  # diffusion is slow, cap runs
        diff_times = []
        horizon = config["data"]["horizon"]
        for i in range(num_diff_runs):
            costmap, start_pos, end_pos = samples[i]
            costmap_t = torch.from_numpy(costmap).float().to(device)
            if costmap_t.dim() == 2:
                costmap_t = costmap_t.unsqueeze(0).unsqueeze(0)
            start_xy = np.array(start_pos[::-1])
            end_xy = np.array(end_pos[::-1])
            norm_start = (start_xy / img_size) * 2 - 1
            norm_end = (end_xy / img_size) * 2 - 1
            start_t = torch.from_numpy(norm_start).float().to(device).unsqueeze(0)
            end_t = torch.from_numpy(norm_end).float().to(device).unsqueeze(0)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = diffusion_scheduler.sample(
                model=model,
                condition=costmap_t,
                shape=(1, horizon, 2),
                start_pos=start_t,
                end_pos=end_t,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            diff_times.append((t1 - t0) * 1000)

        diff_times = np.array(diff_times)
        print(f"\n--- Diffusion inference time ({num_diff_runs} runs, {infer_timesteps} steps) ---")
        print(f"  Mean: {np.mean(diff_times):.4f} ms")
        print(f"  Std:  {np.std(diff_times):.4f} ms")

        print(f"\n--- Summary ---")
        print(f"  A*:       {mean_astar:.4f} ms (mean)")
        print(f"  Diffusion: {np.mean(diff_times):.4f} ms (mean)")
        print(f"  Ratio (Diffusion/A*): {np.mean(diff_times) / mean_astar:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark A* (and optionally Diffusion) inference time.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of costmaps / A* runs")
    parser.add_argument("--diffusion", action="store_true", help="Also run Diffusion and compare")
    args = parser.parse_args()
    main(args)
