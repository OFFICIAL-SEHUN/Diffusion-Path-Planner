import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def visualize_dataset_sample(data_path="data/dataset.pt", save_path="results/dataset_sample.png"):
    """
    Loads the dataset and visualizes a random sample (costmap and path).
    """
    # --- Load Data ---
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print(f"Loading dataset from {data_path}...")
    # Load data onto CPU to avoid CUDA errors if torch was built with it
    data = torch.load(data_path, map_location=torch.device('cpu'))
    costmaps = data["costmaps"]  # Tensor [N, 64, 64]
    paths = data["paths"]        # Tensor [N, H, 2], normalized to [-1, 1]
    print(f"Loaded {len(costmaps)} samples.")

    # --- Select a Random Sample ---
    idx = random.randint(0, len(costmaps) - 1)
    costmap_sample = costmaps[idx].cpu().numpy()
    path_sample_norm = paths[idx].cpu().numpy()

    # --- De-normalize Path from [-1, 1] to pixel coordinates ---
    img_size = costmap_sample.shape[0]
    path_sample_scaled = (path_sample_norm + 1) / 2 * img_size

    # --- Plotting ---
    print(f"Visualizing sample #{idx}")
    plt.figure(figsize=(8, 8))
    
    # Use a colormap similar to the main script for consistency
    cmap = plt.cm.get_cmap('plasma_r').copy()
    cmap.set_bad(color='black')  # Obstacles (inf) will be black
    
    # Mask invalid values (like infinity) for correct color mapping
    masked_costmap = np.ma.masked_invalid(costmap_sample)
    
    plt.imshow(masked_costmap, cmap=cmap, origin='lower', vmin=0.0, vmax=1.0)
    
    # Plot the path
    # path is (row, col) which corresponds to (y, x)
    plt.plot(path_sample_scaled[:, 1], path_sample_scaled[:, 0], 'r-', linewidth=2, label='Path')

    # Mark start and end points
    plt.scatter(path_sample_scaled[0, 1], path_sample_scaled[0, 0], c='purple', marker='o', s=100, label='Start', zorder=5)
    plt.scatter(path_sample_scaled[-1, 1], path_sample_scaled[-1, 0], c='purple', marker='x', s=100, label='End', zorder=5)
    
    plt.legend()
    plt.title(f"Dataset Sample #{idx}")
    plt.xlim(0, img_size)
    plt.ylim(0, img_size)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Create Colorbar explicitly for consistency ---
    norm = plt.Normalize(vmin=0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # This is needed for the mappable to work.
    plt.colorbar(sm, label="Cost")

    # --- Save Figure ---
    results_dir = os.path.dirname(save_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize_dataset_sample()
