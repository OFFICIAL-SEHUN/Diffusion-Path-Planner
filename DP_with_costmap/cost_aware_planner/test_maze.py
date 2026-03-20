"""
Quick test script to visualize maze generation.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from maze import MazeGenerator

# Test different map sizes with appropriate scales
# Keep internal grid ~20-32 for good complexity
test_configs = [
    (32, 4),    # 8x8 grid
    (64, 4),    # 16x16 grid
    (128, 6),   # 21x21 grid
    (256, 10),  # 25x25 grid
]

for img_size, scale in test_configs:
    print(f"Generating {img_size}x{img_size} maze (scale={scale})...")
    
    maze_gen = MazeGenerator(img_size, scale)
    costmap, _, start_pos, end_pos = maze_gen.generate(cost_weight=50.0)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap_vis = cm.get_cmap("plasma_r").copy()
    cmap_vis.set_bad(color="black")
    
    ax.imshow(costmap, cmap=cmap_vis, origin="upper", vmin=0, vmax=1.0)
    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Mark start and end
    ax.plot(start_pos[1], start_pos[0], 'go', markersize=10, label='Start')
    ax.plot(end_pos[1], end_pos[0], 'ro', markersize=10, label='End')
    
    ax.legend(loc="upper right")
    ax.set_title(f"Maze {img_size}x{img_size} (scale={scale})")
    
    save_path = f"test_maze_{img_size}.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

print("\nDone! Check test_maze_*.png files.")
