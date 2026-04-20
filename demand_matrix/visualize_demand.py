"""
Render the demand matrix as a log-scaled heatmap.

Reads  : data/demand_matrix.json
Writes : data/demand_matrix.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"
JSON_PATH = DATA_DIR / "demand_matrix.json"
OUT_PATH  = DATA_DIR / "demand_matrix.png"


def load_matrix(path: Path) -> np.ndarray:
    with open(path) as f:
        raw: dict[str, dict[str, float]] = json.load(f)

    all_keys = sorted(
        {k for k in raw} | {d for dsts in raw.values() for d in dsts},
        key=int,
    )
    n = len(all_keys)
    idx: dict[str, int] = {k: i for i, k in enumerate(all_keys)}

    matrix = np.zeros((n, n), dtype=np.float64)
    for src, dsts in raw.items():
        i = idx[src]
        for dst, val in dsts.items():
            matrix[i, idx[dst]] = val
    return matrix


def render(matrix: np.ndarray, out: Path) -> None:
    log_matrix = np.log1p(matrix)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(log_matrix, aspect="auto", cmap="inferno", interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log(1 + demand)", fontsize=11)

    ax.set_title("Demand Matrix (log scale)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Destination node index", fontsize=11)
    ax.set_ylabel("Origin node index", fontsize=11)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


def main() -> None:
    print(f"Loading {JSON_PATH} …")
    matrix = load_matrix(JSON_PATH)
    print(f"Matrix size: {matrix.shape[0]}×{matrix.shape[1]}")
    render(matrix, OUT_PATH)


if __name__ == "__main__":
    main()
