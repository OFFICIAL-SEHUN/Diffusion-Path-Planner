"""
경로 비교 시각화 모듈
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_comparison(diffusion_result, astar_result, slope_map, height_map, 
                         diffusion_path_pixels, astar_path_pixels, img_size,
                         astar_time=None, diffusion_time=None, patch_result=None, 
                         patch_path_pixels=None, patch_time=None):
    """비교 시각화"""
    
    fig = plt.figure(figsize=(20, 16) if patch_result else (20, 12))
    
    # Slope map 준비
    slope_degrees = np.degrees(slope_map)
    cmap_terrain = plt.colormaps.get_cmap('jet').copy()
    cmap_terrain.set_bad(color='black')
    masked_slope = np.ma.masked_invalid(slope_degrees)
    
    # 1. Slope Map with Paths
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(masked_slope, cmap=cmap_terrain, origin='lower', vmin=0, vmax=35)
    
    # Plot paths
    diffusion_pixels = np.array(diffusion_path_pixels)
    
    # A* 경로 (있으면 표시)
    has_astar = len(astar_path_pixels) > 0
    if has_astar:
        astar_pixels = np.array(astar_path_pixels)
        ax1.plot(astar_pixels[:, 1], astar_pixels[:, 0], 
                'r--', linewidth=3, alpha=0.8, label='A* GT')
    
    ax1.plot(diffusion_pixels[:, 1], diffusion_pixels[:, 0], 
            'c-', linewidth=3, alpha=0.9, label='Diffusion')
    
    if patch_path_pixels:
        patch_pixels = np.array(patch_path_pixels)
        ax1.plot(patch_pixels[:, 1], patch_pixels[:, 0], 
                'g-.', linewidth=3, alpha=0.8, label='Patch Diffusion')
    
    # Start/Goal 마커 (Diffusion 기준)
    ax1.scatter(diffusion_pixels[0, 1], diffusion_pixels[0, 0], 
               c='yellow', marker='o', s=150, edgecolors='black', 
               linewidths=2, zorder=10, label='Start')
    
    ax1.scatter(diffusion_pixels[-1, 1], diffusion_pixels[-1, 0], 
               c='lime', marker='*', s=200, edgecolors='black', 
               linewidths=2, zorder=10, label='Goal')
    
    ax1.set_title('Paths on Slope Map', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.axis('off')
    
    # 2. Height Map with Paths
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(height_map, cmap='terrain', origin='lower')
    
    if has_astar:
        ax2.plot(astar_pixels[:, 1], astar_pixels[:, 0], 
                'r--', linewidth=3, alpha=0.8, label='A* GT')
    
    ax2.plot(diffusion_pixels[:, 1], diffusion_pixels[:, 0], 
            'c-', linewidth=3, alpha=0.9, label='Diffusion')
    
    if patch_path_pixels:
        ax2.plot(patch_pixels[:, 1], patch_pixels[:, 0], 
                'g-.', linewidth=3, alpha=0.8, label='Patch Diffusion')
    
    # Start/Goal 마커 (Height Map에도 동일하게)
    ax2.scatter(diffusion_pixels[0, 1], diffusion_pixels[0, 0], 
               c='yellow', marker='o', s=150, edgecolors='black', 
               linewidths=2, zorder=10, label='Start')
    
    ax2.scatter(diffusion_pixels[-1, 1], diffusion_pixels[-1, 0], 
               c='lime', marker='*', s=200, edgecolors='black', 
               linewidths=2, zorder=10, label='Goal')
    
    ax2.set_title('Paths on Height Map', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.axis('off')
    
    # 3. CoT Cost along Path
    ax3 = plt.subplot(2, 3, 3)
    diffusion_cot_values = [seg['cot_cost'] for seg in diffusion_result['segments']]
    
    if has_astar:
        astar_cot_values = [seg['cot_cost'] for seg in astar_result['segments']]
        ax3.plot(astar_cot_values, 'r-', linewidth=2, label='A* GT', alpha=0.8)
    
    ax3.plot(diffusion_cot_values, 'c-', linewidth=2, label='Diffusion', alpha=0.8)
    
    if patch_result:
        patch_cot_values = [seg['cot_cost'] for seg in patch_result['segments']]
        ax3.plot(patch_cot_values, 'g-.', linewidth=2, label='Patch Diffusion', alpha=0.8)
    
    ax3.set_xlabel('Segment Index', fontsize=12)
    ax3.set_ylabel('CoT Cost', fontsize=12)
    ax3.set_title('CoT Cost Along Path', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Slope Distribution
    ax4 = plt.subplot(2, 3, 4)
    diffusion_slopes = [seg['slope_deg'] for seg in diffusion_result['segments']]
    
    if has_astar:
        astar_slopes = [seg['slope_deg'] for seg in astar_result['segments']]
        ax4.hist(astar_slopes, bins=30, alpha=0.6, color='red', label='A* GT')
    
    ax4.hist(diffusion_slopes, bins=30, alpha=0.6, color='cyan', label='Diffusion')
    
    if patch_result:
        patch_slopes = [seg['slope_deg'] for seg in patch_result['segments']]
        ax4.hist(patch_slopes, bins=30, alpha=0.6, color='green', label='Patch Diffusion')
    
    ax4.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Slope Angle (degrees)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Slope Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Statistics Comparison
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # Build time info text
    time_text = ""
    has_astar = len(astar_path_pixels) > 0
    
    if diffusion_time is not None:
        if has_astar and astar_time is not None:
            speedup = diffusion_time / astar_time if astar_time > 0 else 0
            speed_text = f"A* is {speedup:.2f}x faster" if speedup > 1 else f"Diffusion is {1/speedup:.2f}x faster"
            time_text = f"""
    Inference Time:
      • A* GT:      {astar_time*1000:.2f} ms
      • Diffusion:  {diffusion_time*1000:.2f} ms"""
            if patch_time is not None:
                patch_speedup_vs_astar = patch_time / astar_time if astar_time > 0 else 0
                patch_vs_diffusion = patch_time / diffusion_time if diffusion_time > 0 else 0
                if patch_vs_diffusion > 1:
                    patch_text = f"Patch is {patch_vs_diffusion:.2f}x slower than Diffusion"
                else:
                    patch_text = f"Patch is {1/patch_vs_diffusion:.2f}x faster than Diffusion"
                time_text += f"""
      • Patch:      {patch_time*1000:.2f} ms ({patch_text})"""
            time_text += f"""
      • Speed:      {speed_text}
    """
        else:
            time_text = f"""
    Inference Time:
      • A* GT:      FAILED
      • Diffusion:  {diffusion_time*1000:.2f} ms"""
            if patch_time is not None:
                patch_vs_full = patch_time / diffusion_time if diffusion_time > 0 else 0
                if patch_vs_full > 1:
                    speed_text = f"Patch is {patch_vs_full:.2f}x slower"
                else:
                    speed_text = f"Patch is {1/patch_vs_full:.2f}x faster"
                time_text += f"""
      • Patch:      {patch_time*1000:.2f} ms ({speed_text})"""
            time_text += """
    """
    
    if has_astar and astar_result['total_cot'] > 0:
        cot_text = f"""
    Total CoT Cost:
      • A* GT:      {astar_result['total_cot']:.2f}
      • Diffusion:  {diffusion_result['total_cot']:.2f}
      • Ratio:      {diffusion_result['total_cot'] / astar_result['total_cot']:.2f}x"""
        
        if patch_result:
            cot_text += f"""
      • Patch:      {patch_result['total_cot']:.2f}
      • Ratio:      {patch_result['total_cot'] / astar_result['total_cot']:.2f}x"""
    else:
        cot_text = f"""
    Total CoT Cost:
      • A* GT:      FAILED
      • Diffusion:  {diffusion_result['total_cot']:.2f}"""
        
        if patch_result:
            cot_text += f"""
      • Patch:      {patch_result['total_cot']:.2f}"""
    
    stats_text = f"""
    ╔══════════════════════════════════════╗
    ║       PATH COMPARISON RESULTS        ║
    ╚══════════════════════════════════════╝
    {time_text}{cot_text}
    
    Average CoT:
      • A* GT:      {astar_result['avg_cot']:.3f}
      • Diffusion:  {diffusion_result['avg_cot']:.3f}"""
    
    if patch_result:
        stats_text += f"""
      • Patch:      {patch_result['avg_cot']:.3f}"""
    
    stats_text += f"""
    
    Path Length:
      • A* GT:      {astar_result['stats']['total_distance']:.1f}m
      • Diffusion:  {diffusion_result['stats']['total_distance']:.1f}m"""
    
    if patch_result:
        stats_text += f"""
      • Patch:      {patch_result['stats']['total_distance']:.1f}m"""
    
    slope_stats_line2 = f" / {patch_result['stats']['avg_slope']:6.2f}°" if patch_result else ""
    slope_stats_line3 = f" / {patch_result['stats']['max_slope']:6.2f}°" if patch_result else ""
    slope_stats_line4 = f" / {patch_result['stats']['min_slope']:6.2f}°" if patch_result else ""
    
    method_label = "A* / Diffusion / Patch" if patch_result else "A* / Diffusion"
    
    stats_text += f"""
    
    Slope Statistics ({method_label}):
      • Avg:  {astar_result['stats']['avg_slope']:6.2f}° / {diffusion_result['stats']['avg_slope']:6.2f}°{slope_stats_line2}
      • Max:  {astar_result['stats']['max_slope']:6.2f}° / {diffusion_result['stats']['max_slope']:6.2f}°{slope_stats_line3}
      • Min:  {astar_result['stats']['min_slope']:6.2f}° / {diffusion_result['stats']['min_slope']:6.2f}°{slope_stats_line4}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax5.transAxes)
    
    # 6. Uphill/Downhill CoT Comparison
    ax6 = plt.subplot(2, 3, 6)
    categories = ['Uphill\nCoT', 'Downhill\nCoT', 'Total\nCoT']
    diffusion_values = [diffusion_result['stats']['uphill_cot'],
                       diffusion_result['stats']['downhill_cot'],
                       diffusion_result['total_cot']]
    
    x = np.arange(len(categories))
    
    if has_astar:
        astar_values = [astar_result['stats']['uphill_cot'], 
                        astar_result['stats']['downhill_cot'],
                        astar_result['total_cot']]
        width = 0.35
        ax6.bar(x - width/2, astar_values, width, label='A* GT', color='red', alpha=0.7)
        ax6.bar(x + width/2, diffusion_values, width, label='Diffusion', color='cyan', alpha=0.7)
    else:
        width = 0.5
        ax6.bar(x, diffusion_values, width, label='Diffusion', color='cyan', alpha=0.7)
    
    ax6.set_ylabel('CoT Cost', fontsize=12)
    ax6.set_title('CoT Breakdown', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(categories)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    if has_astar:
        title = 'Diffusion vs A* Path Comparison'
    else:
        title = 'Diffusion Path Planning (A* Failed - Terrain Too Complex)'
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'path_cost_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()
