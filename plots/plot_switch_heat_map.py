"""
Paper-ready heatmap of switch states over time.

Input:
    results/switch_states.csv
Output:
    figures/switch_heatmap.pdf
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Global plotting settings
# =========================
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "switch_states_GA_H0_L02.csv"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


# =========================
# Load data
# =========================
df = pd.read_csv(RESULTS)

required_cols = {"time", "switch", "state"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns {required_cols}")


# =========================
# Pivot table
# =========================
pivot = df.pivot(
    index="switch",
    columns="time",
    values="state"
).sort_index()

# Optional: keep only switches that change at least once
# active = pivot.diff(axis=1).abs().sum(axis=1) > 0
changes = pivot.diff(axis=1).abs().fillna(0)
# pivot = pivot.loc[changes]


# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(7.0, 3.5))  # 1-column paper width

im = ax.imshow(
    changes.values,
    aspect="auto",
    interpolation="nearest",
    cmap="Greys",
    vmin=0,
    vmax=1
)

# Axes labels
ax.set_xlabel("Time [h]")
ax.set_ylabel("Switch")

# X ticks (clean & sparse)
n_ticks = 12
x_positions = np.linspace(0, pivot.shape[1] - 1, n_ticks)
x_labels = np.linspace(pivot.columns.min(), pivot.columns.max(), n_ticks)

ax.set_xticks(x_positions)
ax.set_xticklabels(np.round(x_labels, 1))

# Y ticks
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)

# Colorbar (discrete)
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_ticks([0, 1])
cbar.set_label("Switch Change (0=same, 1=alternate)")

# Layout & save
plt.tight_layout()
output = FIGURES / "switch_heatmap_GA_H0_L02.pdf"
plt.savefig(output)
plt.close()


print(f"Saved figure to {output}")
