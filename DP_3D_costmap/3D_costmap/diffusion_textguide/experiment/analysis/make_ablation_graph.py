import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. CSV log paths
# =========================
log_files = {
    "ResNet-18 (P)": "backbone_ablation_resnet_20260513_132630.csv",
    "ResNet-18 (S)": "backbone_ablation_resnet_20260514_091833.csv",
    # "ConvNeXt-T (P)": "backbone_ablation_convnext_20260513_132703.csv",
}

# CSV 파일들이 있는 폴더
log_dir = Path("/workspace/diffusion_textguide/results/backbone_ablation_10intent")   # 필요하면 "/mnt/data" 등으로 수정

# =========================
# 2. Load and plot val_loss
# =========================
plt.figure(figsize=(9, 5.2))

summary = []

for model_name, file_name in log_files.items():
    csv_path = log_dir / file_name
    df = pd.read_csv(csv_path)

    # 필요한 column 확인
    if "epoch" not in df.columns or "val_loss" not in df.columns:
        raise ValueError(f"{file_name} must contain 'epoch' and 'val_loss' columns.")

    # val_loss가 존재하는 row만 사용
    df_val = df[["epoch", "val_loss"]].dropna().copy()

    # curve plot
    plt.plot(
        df_val["epoch"],
        df_val["val_loss"],
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        label=model_name,
    )

    # final / best val_loss 요약 저장
    final_row = df_val.iloc[-1]
    best_row = df_val.loc[df_val["val_loss"].idxmin()]

    summary.append({
        "Model": model_name,
        "Final Epoch": int(final_row["epoch"]),
        "Final Val Loss": float(final_row["val_loss"]),
        "Best Epoch": int(best_row["epoch"]),
        "Best Val Loss": float(best_row["val_loss"]),
    })

# =========================
# 3. Figure style
# =========================
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curve: 5k Backbone Ablation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# =========================
# 4. Save figure
# =========================
plt.savefig("val_loss_curve_5k_backbone_ablation.png", dpi=300)
plt.savefig("val_loss_curve_5k_backbone_ablation.pdf")

plt.show()

# =========================
# 5. Print summary table
# =========================
summary_df = pd.DataFrame(summary)
print(summary_df)

# 필요하면 CSV로 저장
# summary_df.to_csv("val_loss_summary_5k_backbone_ablation.csv", index=False)