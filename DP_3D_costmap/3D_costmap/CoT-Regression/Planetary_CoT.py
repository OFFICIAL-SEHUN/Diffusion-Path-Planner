"""
CoT vs slope: Isaac Lab measured points + 4th-degree polynomial fit.

Exports ``cot_model.json`` for ``diffusion_textguide/scripts/generate_data.py``.
Run from repo root (with numpy/matplotlib):

    python3 DP_3D_costmap/3D_costmap/CoT-Regression/Planetary_CoT.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

# Table 2 — Isaac Lab Go2 (35° trained policy, 1.0 m/s, 30 m runs)
MEASURED_X = np.array([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20], dtype=np.float64)
MEASURED_Y = np.array(
    [0.7400, 0.6803, 0.5805, 0.4921, 0.4201, 0.4324, 0.4848, 0.7792, 0.7246, 1.3935],
    dtype=np.float64,
)

# np.polyfit returns highest-degree first: p[0]*x^4 + p[1]*x^3 + ... + p[4]
_POLY_COEFFS_DESC = np.polyfit(MEASURED_X, MEASURED_Y, 4)


def _configure_plot_font() -> None:
    """Use Times New Roman for all figure text (serif fallback on Linux/Docker)."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" in available:
        family = "Times New Roman"
    else:
        family = next(
            (
                name
                for name in (
                    "Times",
                    "Nimbus Roman",
                    "Liberation Serif",
                    "STIXGeneral",
                    "STIX Two Text",
                    "DejaVu Serif",
                )
                if name in available
            ),
            "serif",
        )
        print(f"[Font] 'Times New Roman' not installed; using {family}")

    plt.rcParams.update(
        {
            "font.family": family,
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Liberation Serif",
                "STIXGeneral",
                "STIX Two Text",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
        }
    )


def calculate_paper_cot(slope_deg):
    """4th-degree polynomial CoT(s), s in degrees (same as coefficient.py / polyfit)."""
    return np.polyval(_POLY_COEFFS_DESC, np.asarray(slope_deg, dtype=np.float64))


def export_cot_model_json(slope_limit_deg: float = 25.0) -> Path:
    """Write ``cot_model.json`` using the module-level polyfit (same as ``calculate_paper_cot``)."""
    p = _POLY_COEFFS_DESC
    a, b, c, d, e = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])

    grid = np.linspace(-slope_limit_deg, slope_limit_deg, 257)
    vals = a * grid**4 + b * grid**3 + c * grid**2 + d * grid + e
    g_min = float(vals.min())
    g_max = float(vals.max())

    out_dir = Path(__file__).resolve().parents[1] / "diffusion_textguide" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cot_model.json"

    payload = {
        "model": "poly4_deg",
        "description": (
            "Isaac Lab Unitree Go2 — CoT vs slope (measured Table 2), "
            "4th-degree polynomial via np.polyfit (CoT-Regression)."
        ),
        "coefficients": {"a": a, "b": b, "c": c, "d": d, "e": e},
        "normalization": {
            "slope_limit_deg": float(slope_limit_deg),
            "g_min": g_min,
            "g_max": g_max,
        },
        "measurements_deg": MEASURED_X.tolist(),
        "measurements_cot": MEASURED_Y.tolist(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return out_path


# --- plot (same as before, uses current polyfit) ---
if __name__ == "__main__":
    export_cot_model_json()
    _configure_plot_font()

    x_range = np.linspace(-25, 20, 400)
    y_cot = calculate_paper_cot(x_range)

    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_cot, color="red", linewidth=4, label="Fitted CoT model")
    plt.scatter(MEASURED_X, MEASURED_Y, color="red", marker="^", s=80, label="Measured CoT")
    plt.title("Slope vs CoT", fontsize=27)
    plt.xlabel("Slope angle (deg)", fontsize=21)
    plt.ylabel("Cost of transport", fontsize=21)
    plt.xticks(np.arange(-25, 21, 5), fontsize=21)
    plt.yticks(fontsize=21)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=15)

    out_path = Path(__file__).resolve().parent / "cot_plot.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")
