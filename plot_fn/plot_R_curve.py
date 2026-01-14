import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_R_curve(self_R_curve):

    R = self_R_curve
    if not R:
        return

    os.makedirs("results", exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    times = np.array(sorted(R.keys()), dtype=float)   # horas (0..23)
    values = np.array([float(R[t]) for t in times], dtype=float)

    # opcional: pérdida de resiliencia (área)
    dt = 1.0  # 1h
    R_loss = float(np.sum((1.0 - values) * dt))

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.plot(times, values, marker="o", markersize=3, linewidth=1.4)

    ax.set_title(f"Resilience curve R(t)  (loss = {R_loss:.2f} h)")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("R(t) = served/expected")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(times.min(), times.max())
    ax.grid(alpha=0.25)

    # sombreado opcional (muy visual, pero sobrio)
    ax.fill_between(times, values, 1.0, alpha=0.12)

    fig.tight_layout()
    fig.savefig("results/R_curve.pdf", bbox_inches="tight")
    fig.savefig("results/R_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Guardada R_curve en results/R_curve.pdf / .png")