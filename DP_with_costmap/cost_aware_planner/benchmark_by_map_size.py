"""
Map-size-wise inference time benchmark: A* vs Diffusion.
1) Generate maps per size
2) Measure inference time (A* and Diffusion) per size
3) Write metrics/table (CSV + Markdown)
4) Optionally draw costmap + paths (A*, Dijkstra, RRT*, Diffusion) per size

python benchmark_by_map_size.py --map_sizes 32 64 --planners astar,dijkstra,rrt_star --diffusion --draw_map
"""
import argparse
import gc
import os
import random
import time
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Avoid GUI backend to reduce memory (prevents OOM on headless/low-RAM)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from maze import MazeGenerator, a_star_search
from pathfinding import dijkstra_search, rrt_star_search


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_planner_benchmark(samples, cost_weight, num_runs, planner="astar"):
    """Run one planner on each sample, return times in ms."""
    times = []
    for costmap, start_pos, end_pos in samples[:num_runs]:
        start_pos = tuple(start_pos)
        end_pos = tuple(end_pos)
        t0 = time.perf_counter()
        if planner == "astar":
            a_star_search(costmap, start_pos, end_pos, cost_weight=cost_weight)
        elif planner == "dijkstra":
            dijkstra_search(costmap, start_pos, end_pos, cost_weight=cost_weight)
        elif planner == "rrt_star":
            rrt_star_search(costmap, start_pos, end_pos, cost_weight=cost_weight)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.array(times)


def run_diffusion_benchmark(samples, config, model, diffusion_scheduler, device, img_size, num_runs):
    """Run Diffusion sampling on each sample, return times in ms."""
    import torch
    horizon = config["data"]["horizon"]
    times = []
    for i in range(min(len(samples), num_runs)):
        costmap, start_pos, end_pos = samples[i]
        costmap_t = torch.from_numpy(costmap).float().to(device)
        if costmap_t.dim() == 2:
            costmap_t = costmap_t.unsqueeze(0).unsqueeze(0)
        start_xy = np.array(start_pos[::-1])
        end_xy = np.array(end_pos[::-1])
        norm_start = (start_xy / img_size) * 2 - 1
        norm_end = (end_xy / img_size) * 2 - 1
        start_t = torch.from_numpy(norm_start.astype(np.float32)).float().to(device).unsqueeze(0)
        end_t = torch.from_numpy(norm_end.astype(np.float32)).float().to(device).unsqueeze(0)

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
        times.append((t1 - t0) * 1000)
    return np.array(times)


def draw_map_and_paths(costmap, paths, img_size, save_path):
    """
    paths: dict planner_name -> path (list of (row,col) or Nx2 array).
    Costmap [H,W]; plot path as (col, row) for imshow origin='upper'.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = cm.get_cmap("plasma_r").copy()
    cmap.set_bad(color="black")
    ax.imshow(costmap, cmap=cmap, origin="upper", vmin=0, vmax=1.0)
    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)
    ax.grid(True, linestyle="--", alpha=0.5)

    # A*: red dashed line, Diffusion: solid black line
    styles = {
        "astar": {"color": "red", "linestyle": "--", "linewidth": 2.5, "alpha": 0.9},
        "diffusion": {"color": "black", "linestyle": "-", "linewidth": 2, "alpha": 0.9}
    }
    for name, path in paths.items():
        if path is None or len(path) == 0:
            continue
        path = np.array(path)
        if path.ndim == 1:
            continue
        x, y = path[:, 1], path[:, 0]  # col, row
        label = {"astar": "A*", "diffusion": "Diffusion"}.get(name, name)
        style = styles.get(name, {"color": "gray", "linestyle": "-", "linewidth": 2, "alpha": 0.9})
        ax.plot(x, y, label=label, **style)

    ax.legend(loc="upper right")
    ax.set_title(f"Costmap {img_size}x{img_size} & Paths")
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    cost_weight = config["data"]["cost_weight"]
    scale = config["maze"]["scale"]
    map_sizes = args.map_sizes  # e.g. [32, 64, 128, 256]
    num_runs_astar = args.num_runs_astar
    num_runs_diffusion = args.num_runs_diffusion
    planners = [p.strip().lower() for p in args.planners.split(",") if p.strip()]
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Diffusion loaded lazily (after RRT*) to avoid OOM: RRT* can use a lot of RAM.
    run_diffusion = args.diffusion
    model = None
    diffusion_scheduler = None
    device = "cpu"
    num_runs_rrt = args.num_runs_rrt_star
    model_path = None
    if run_diffusion:
        model_path = os.path.join(
            config["training"]["checkpoint_dir"],
            config["training"]["model_name"],
        )
        if not os.path.isfile(model_path):
            print(f"Model not found: {model_path}. Skipping Diffusion benchmark.")
            run_diffusion = False

    # Map-size-specific scale: keep internal grid ~20-32 cells for complexity
    # Internal grid = img_size // scale, so scale = img_size // target_grid_size
    scale_by_size = {
        32: 4,    # 32/4 = 8x8 grid
        64: 4,    # 64/4 = 16x16 grid
        128: 6,   # 128/6 = ~21x21 grid
        256: 10,  # 256/10 = ~25x25 grid
        512: 16   # 512/16 = 32x32 grid
    }

    results = []
    for img_size in map_sizes:
        # Choose scale: use map-specific or fallback to config default
        map_scale = scale_by_size.get(img_size, max(4, img_size // 20))
        print(f"\n--- Map size {img_size}x{img_size} (scale={map_scale}, grid={img_size//map_scale}x{img_size//map_scale}) ---")
        maze_gen = MazeGenerator(img_size, map_scale)
        samples = []
        n_gen = max(num_runs_astar, num_runs_diffusion)
        for _ in range(n_gen):
            costmap, _, start_pos, end_pos = maze_gen.generate(cost_weight=cost_weight)
            if isinstance(costmap, np.ndarray):
                costmap = np.where(np.isinf(costmap), 1.0, costmap).astype(np.float32)
            samples.append((costmap, tuple(start_pos), tuple(end_pos)))

        # A* timing
        a_star_times = run_planner_benchmark(samples, cost_weight, num_runs_astar, "astar")
        mean_astar = float(np.mean(a_star_times))
        std_astar = float(np.std(a_star_times))
        print(f"  A*: mean={mean_astar:.2f} ms, std={std_astar:.2f} ms (n={num_runs_astar})")

        # Skip other planners - only compare A* vs Diffusion
        mean_dijkstra = std_dijkstra = None
        mean_rrt_star = std_rrt_star = None

        gc.collect()  # Free memory before loading Diffusion model

        # Lazy-load Diffusion model (after RRT*) to avoid OOM
        if run_diffusion and model is None and model_path:
            os.environ["TQDM_DISABLE"] = "1"
            import torch
            from model import ConditionalPathModel
            from diffusion import DiffusionScheduler
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Loading Diffusion model (device: {device})...")
            model = ConditionalPathModel(config=config)
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
            print(f"  Diffusion timesteps: {infer_timesteps}")

        # Diffusion timing (model accepts arbitrary spatial size via AdaptiveAvgPool2d)
        mean_diffusion = std_diffusion = None
        if run_diffusion and model is not None and diffusion_scheduler is not None:
            try:
                diff_times = run_diffusion_benchmark(
                    samples, config, model, diffusion_scheduler,
                    device, img_size, num_runs_diffusion
                )
                mean_diffusion = float(np.mean(diff_times))
                std_diffusion = float(np.std(diff_times))
                print(f"  Diffusion: mean={mean_diffusion:.2f} ms, std={std_diffusion:.2f} ms (n={num_runs_diffusion})")
            except Exception as e:
                print(f"  Diffusion: failed ({e})")
                mean_diffusion = std_diffusion = None

        results.append({
            "map_size": img_size,
            "a_star_mean_ms": mean_astar,
            "a_star_std_ms": std_astar,
            "dijkstra_mean_ms": mean_dijkstra,
            "dijkstra_std_ms": std_dijkstra,
            "rrt_star_mean_ms": mean_rrt_star,
            "rrt_star_std_ms": std_rrt_star,
            "diffusion_mean_ms": mean_diffusion,
            "diffusion_std_ms": std_diffusion,
        })

        # --- Draw map + paths (first sample) ---
        if getattr(args, "draw_map", False):
            costmap, start_pos, end_pos = samples[0]
            start_pos = tuple(start_pos)
            end_pos = tuple(end_pos)
            paths = {}
            path_astar = a_star_search(costmap, start_pos, end_pos, cost_weight=cost_weight)
            paths["astar"] = path_astar
            # Only compare A* vs Diffusion
            if run_diffusion and model is not None and diffusion_scheduler is not None:
                import torch
                costmap_t = torch.from_numpy(costmap).float().to(device).unsqueeze(0).unsqueeze(0)
                start_xy = np.array(start_pos[::-1])
                end_xy = np.array(end_pos[::-1])
                norm_start = (start_xy / img_size) * 2 - 1
                norm_end = (end_xy / img_size) * 2 - 1
                start_t = torch.from_numpy(norm_start.astype(np.float32)).float().to(device).unsqueeze(0)
                end_t = torch.from_numpy(norm_end.astype(np.float32)).float().to(device).unsqueeze(0)
                with torch.no_grad():
                    gen = diffusion_scheduler.sample(
                        model=model,
                        condition=costmap_t,
                        shape=(1, config["data"]["horizon"], 2),
                        start_pos=start_t,
                        end_pos=end_t,
                    )
                gen_np = gen.squeeze().cpu().numpy()
                path_diffusion = (gen_np + 1) / 2 * img_size  # [H, 2] row,col
                paths["diffusion"] = path_diffusion
            map_path = os.path.join(out_dir, f"map_{img_size}.png")
            draw_map_and_paths(costmap, paths, img_size, map_path)
            print(f"  Map saved: {map_path}")

    # --- Write metrics table (A* vs Diffusion only) ---
    csv_path = os.path.join(out_dir, "inference_time_by_map_size.csv")
    with open(csv_path, "w") as f:
        f.write("map_size,a_star_mean_ms,a_star_std_ms,diffusion_mean_ms,diffusion_std_ms,ratio_astar_diff\n")
        for r in results:
            ratio = ""
            if r["diffusion_mean_ms"] is not None and r["diffusion_mean_ms"] > 0:
                ratio = f"{r['a_star_mean_ms'] / r['diffusion_mean_ms']:.2f}"
            def _v(v):
                return f"{v:.4f}" if v is not None else "N/A"
            f.write(f"{r['map_size']},{r['a_star_mean_ms']:.4f},{r['a_star_std_ms']:.4f},{_v(r['diffusion_mean_ms'])},{_v(r['diffusion_std_ms'])},{ratio}\n")
    print(f"\nWrote {csv_path}")

    md_path = os.path.join(out_dir, "inference_time_by_map_size.md")
    with open(md_path, "w") as f:
        f.write("# Inference Time by Map Size (A* vs Diffusion)\n\n")
        f.write("| Map Size | A* (ms) | Diffusion (ms) | A*/Diff |\n")
        f.write("|----------|---------|----------------|--------|\n")
        def _m(r, k):
            return f"{r[k]:.2f}" if r.get(k) is not None else "N/A"
        for r in results:
            ratio = ""
            if r["diffusion_mean_ms"] is not None and r["diffusion_mean_ms"] > 0:
                ratio = f"{r['a_star_mean_ms'] / r['diffusion_mean_ms']:.2f}x"
            f.write(f"| {r['map_size']} | {_m(r,'a_star_mean_ms')} | {_m(r,'diffusion_mean_ms')} | {ratio} |\n")
        f.write("\n- **A***: Cost-aware A* planner.\n- **Diffusion**: Neural diffusion-based planner (GPU/CPU).\n- **A*/Diff**: Speed ratio (A* time / Diffusion time).\n")
    print(f"Wrote {md_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark A* vs Diffusion inference time by map size.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--map_sizes", type=int, nargs="+", default=[32, 64, 128, 256],
                        help="Map sizes (e.g. 32 64 128 256)")
    parser.add_argument("--num_runs_astar", type=int, default=20, help="Number of planner runs per map size")
    parser.add_argument("--num_runs_diffusion", type=int, default=5, help="Number of Diffusion runs per map size")
    parser.add_argument("--num_runs_rrt_star", type=int, default=5, help="Number of RRT* runs (fewer to reduce memory)")
    parser.add_argument("--planners", type=str, default="astar", help="[Deprecated] Only A* vs Diffusion comparison")
    parser.add_argument("--output_dir", type=str, default="metrics", help="Directory for CSV and Markdown table")
    parser.add_argument("--diffusion", action="store_true", help="Include Diffusion benchmark (requires trained model)")
    parser.add_argument("--draw_map", action="store_true", help="Draw costmap + paths (A*, Dijkstra, RRT*, Diffusion) per map size → metrics/map_<size>.png")
    args = parser.parse_args()
    main(args)
