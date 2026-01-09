# """
# Paper-ready heatmap of switch states over time.

# Input:
#     results/switch_states.csv
# Output:
#     figures/switch_heatmap.pdf
# """

# import pandas as pd
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from pathlib import Path


# # =========================
# # Global plotting settings
# # =========================
# mpl.rcParams.update({
#     "font.family": "serif",
#     "font.size": 9,
#     "axes.labelsize": 9,
#     "axes.titlesize": 9,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "legend.fontsize": 8,
#     "lines.linewidth": 1.2,
#     "axes.linewidth": 0.8,
#     "grid.linewidth": 0.4,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })


# # =========================
# # Paths
# # =========================
# ROOT = Path(__file__).resolve().parents[1]
# RESULTS = ROOT / "results" / "switch_states_GA_PRUEBA.csv"
# FIGURES = ROOT / "figures"
# FIGURES.mkdir(exist_ok=True)


# # =========================
# # Load data
# # =========================
# df = pd.read_csv(RESULTS)

# required_cols = {"time", "switch", "state"}
# if not required_cols.issubset(df.columns):
#     raise ValueError(f"CSV must contain columns {required_cols}")


# # =========================
# # Pivot table
# # =========================
# pivot = df.pivot(
#     index="switch",
#     columns="time",
#     values="state"
# ).sort_index()

# # Optional: keep only switches that change at least once
# # active = pivot.diff(axis=1).abs().sum(axis=1) > 0
# changes = pivot.diff(axis=1).abs().fillna(0)
# # pivot = pivot.loc[changes]


# # =========================
# # Plot
# # =========================
# fig, ax = plt.subplots(figsize=(7.0, 3.5))  # 1-column paper width

# im = ax.imshow(
#     changes.values,
#     aspect="auto",
#     interpolation="nearest",
#     cmap="Greys",
#     vmin=0,
#     vmax=1
# )

# # Axes labels
# ax.set_xlabel("Time [h]")
# ax.set_ylabel("Switch")

# # X ticks (clean & sparse)
# n_ticks = 12
# min_time : int = int(pivot.columns.min())
# max_time : int = int(pivot.columns.max())
# x_positions = np.linspace(0, pivot.shape[1] - 1, n_ticks)
# x_labels = np.linspace(min_time, max_time, n_ticks)

# ax.set_xticks(x_positions)
# ax.set_xticklabels(np.round(x_labels, 1))

# # Y ticks
# ax.set_yticks(range(len(pivot.index)))
# ax.set_yticklabels(pivot.index)

# # Colorbar (discrete)
# cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
# cbar.set_ticks([0, 1])
# cbar.set_label("Switch Change (0=same, 1=alternate)")

# # Layout & save
# plt.tight_layout()
# output = FIGURES / "switch_heatmap_GA_PRUEBA.pdf"
# plt.savefig(output)
# plt.close()


# print(f"Saved figure to {output}")

# """
# Paper-ready heatmap of switch states over time.

# Input:
#     results/switch_states.csv
# Output:
#     figures/switch_heatmap.pdf
# """

# import pandas as pd
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from pathlib import Path


# # =========================
# # Global plotting settings
# # =========================
# mpl.rcParams.update({
#     "font.family": "serif",
#     "font.size": 9,
#     "axes.labelsize": 9,
#     "axes.titlesize": 9,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "legend.fontsize": 8,
#     "lines.linewidth": 1.2,
#     "axes.linewidth": 0.8,
#     "grid.linewidth": 0.4,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })


# # =========================
# # Paths
# # =========================
# ROOT = Path(__file__).resolve().parents[1]
# RESULTS = ROOT / "results" / "switch_states_GA_PRUEBA.csv"
# FIGURES = ROOT / "figures"
# FIGURES.mkdir(exist_ok=True)


# # =========================
# # Load data
# # =========================
# df = pd.read_csv(RESULTS)

# required_cols = {"time", "switch", "state", "sw_in_service"}
# if not required_cols.issubset(df.columns):
#     raise ValueError(f"CSV must contain columns {required_cols}")


# # =========================
# # Pivot tables
# # =========================
# # Estado del switch
# pivot_state = df.pivot(
#     index="switch",
#     columns="time",
#     values="state"
# ).sort_index()

# # Estado de la línea asociada
# pivot_line = df.pivot(
#     index="switch",
#     columns="time",
#     values="sw_in_service"
# ).sort_index()

# # Cambios de switch (0/1)
# changes = pivot_state.diff(axis=1).abs().fillna(0)

# # Máscara: cambio de switch CUANDO la línea está fuera de servicio
# failed_mask = (pivot_line == 0)

# # =========================
# # Plot
# # =========================
# fig, ax = plt.subplots(figsize=(7.0, 3.5))  # 1-column paper width

# # Base heatmap: cambios normales (gris)
# im = ax.imshow(
#     changes.values,
#     aspect="auto",
#     interpolation="nearest",
#     cmap="Greys",
#     vmin=0,
#     vmax=1
# )

# # Overlay rojo: cambios con línea fuera de servicio
# ax.imshow(
#     failed_mask.values,
#     aspect="auto",
#     interpolation="nearest",
#     cmap=mpl.colors.ListedColormap(["none", "red"]),
#     vmin=0,
#     vmax=1,
#     alpha=0.8
# )

# # Axes labels
# ax.set_xlabel("Time [h]")
# ax.set_ylabel("Switch")

# # X ticks (clean & sparse)
# n_ticks = 12
# min_time : int = int(changes.columns.min())
# max_time : int = int(changes.columns.max())
# x_positions = np.linspace(0, changes.shape[1] - 1, n_ticks)
# x_labels = np.linspace(min_time, max_time, n_ticks)

# ax.set_xticks(x_positions)
# ax.set_xticklabels(np.round(x_labels, 1))

# # Y ticks
# ax.set_yticks(range(len(changes.index)))
# ax.set_yticklabels(changes.index)

# # Colorbar (discrete)
# cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
# cbar.set_ticks([0, 1])
# cbar.set_label("Switch Change (0=same, 1=alternate)")

# # Layout & save
# plt.tight_layout()
# output = FIGURES / "switch_heatmap_GA_PRUEBA.pdf"
# plt.savefig(output)
# plt.close()


# print(f"Saved figure to {output}")

"""
Paper-ready heatmap of switch states over time.

Colors:
    White = switch closed
    Black = switch open
    Red   = associated line out of service (overrides switch state)

Input:
    results/switch_states_GA_PRUEBA.csv
Output:
    figures/switch_heatmap_GA_PRUEBA.pdf
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
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
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "switch_states_GA_PRUEBA.csv"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


# =========================
# Load data
# =========================
df = pd.read_csv(RESULTS)

required_cols = {"time", "switch", "state", "sw_in_service"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV must contain columns {required_cols}. Missing: {missing}")

# Asegurar tipos (por si vienen como strings/bools)
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["switch"] = pd.to_numeric(df["switch"], errors="coerce")
df["state"] = pd.to_numeric(df["state"], errors="coerce").fillna(0).astype(int)
df["sw_in_service"] = pd.to_numeric(df["sw_in_service"], errors="coerce").fillna(1).astype(int)

# Orden
df = df.sort_values(["switch", "time"])


# =========================
# Pivot tables
# =========================
pivot_state = df.pivot(index="switch", columns="time", values="state").sort_index()
pivot_line  = df.pivot(index="switch", columns="time", values="sw_in_service").sort_index()

# Asegurar que columnas (tiempos) están ordenadas
pivot_state = pivot_state.reindex(sorted(pivot_state.columns), axis=1)
pivot_line  = pivot_line.reindex(sorted(pivot_line.columns), axis=1)

# Relleno por si faltan timesteps (mantener último valor conocido)
pivot_state = pivot_state.ffill(axis=1).bfill(axis=1)
pivot_line  = pivot_line.ffill(axis=1).bfill(axis=1)

# =========================
# Build heatmap matrix with priority:
#   2 = out of service (red)
#   1 = closed (white)
#   0 = open (black)
# =========================
Z = pivot_state.copy()

# 0=closed, 1=open (si tu "state" ya es 0/1 esto encaja)
Z = 1 - Z.astype(int)

# Override: line out of service -> 2 (red)
Z[pivot_line.astype(int) == 0] = 2
Z_df = Z              # DataFrame
Z_array = Z.to_numpy()
# Z = Z.values


# =========================
# Plot (discrete colormap)
# =========================
fig, ax = plt.subplots(figsize=(7.0, 3.2))  # compact 1-column

cmap = ListedColormap(["white", "black", "red"])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

im = ax.imshow(
    Z_array,
    aspect="auto",
    interpolation="nearest",
    cmap=cmap,
    norm=norm
)

# Axes labels
ax.set_xlabel("Time [h]")
ax.set_ylabel("Switch")

# X ticks (clean & sparse)
times = pivot_state.columns.to_numpy()
n_ticks = min(12, len(times))
x_positions = np.linspace(0, len(times) - 1, n_ticks)
x_labels = np.interp(x_positions, np.arange(len(times)), times)

ax.set_xticks(x_positions)
ax.set_xticklabels(np.round(x_labels, 1))

# Y ticks (si hay muchos switches, mejor sparse)
switch_ids = pivot_state.index.to_numpy()
max_yticks = 25
if len(switch_ids) <= max_yticks:
    ax.set_yticks(np.arange(len(switch_ids)))
    ax.set_yticklabels(switch_ids.astype(int))
else:
    step = int(np.ceil(len(switch_ids) / max_yticks))
    yt = np.arange(0, len(switch_ids), step)
    ax.set_yticks(yt)
    ax.set_yticklabels(switch_ids[yt].astype(int))

# Thin frame
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

# Legend (better than colorbar for discrete classes)
legend_handles = [
    Patch(facecolor="white", edgecolor="black", label="Switch closed"),
    Patch(facecolor="black", edgecolor="black", label="Switch open"),
    Patch(facecolor="red", edgecolor="black", label="Line out of service"),
]
ax.legend(
    handles=legend_handles,
    loc="upper right",
    frameon=True,
    framealpha=0.95
)

plt.tight_layout()
output = FIGURES / "switch_heatmap_GA_PRUEBA.pdf"
plt.savefig(output)
plt.close()

print(f"Saved figure to {output}")
