import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandapower.topology as top
from matplotlib.lines import Line2D

def plot_network(self_net, self_last_gust_speed, self_grid_x, self_grid_y, self_line_status, hour):
    # -------------------------
    # Paper-like rcParams (local)
    # -------------------------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    os.makedirs("figures", exist_ok=True)

    net = self_net
    fig, ax = plt.subplots(figsize=(7.2, 5.2))  # buen tamaño 1-col / 1.5-col

    # =======================================
    #  GRAFO REAL (respeta switches)
    # =======================================
    G = top.create_nxgraph(net, respect_switches=True, include_out_of_service=False)

    # =======================================
    # POSICIONES (km y centrado)
    # =======================================
    FT_TO_KM = 0.0003048
    pos_km = {}
    for bus in net.bus.index:
        x = float(net.bus.at[bus, "x"]) * FT_TO_KM
        y = float(net.bus.at[bus, "y"]) * FT_TO_KM
        pos_km[bus] = (x, y)

    xs = np.array([p[0] for p in pos_km.values()])
    ys = np.array([p[1] for p in pos_km.values()])
    x_mean, y_mean = xs.mean(), ys.mean()
    pos_km = {b: (pos_km[b][0] - x_mean, pos_km[b][1] - y_mean) for b in pos_km}

    # Solo nodos presentes en el grafo
    nodes = list(G.nodes())
    pos = {n: pos_km[n] for n in nodes}

    # =======================================
    # 1) VIENTO (fondo)
    # =======================================
    X, Y = np.meshgrid(self_grid_x, self_grid_y)

    im = ax.imshow(
        self_last_gust_speed,
        origin="lower",
        cmap="coolwarm",
        alpha=0.2,  # más suave que antes
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        vmin=15, vmax=30,
        zorder=0
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Gust speed [m/s]")
    cbar.ax.tick_params(labelsize=7)

    # =======================================
    # 2) BUSES energizados vs no
    # =======================================
    slack_bus = int(net.ext_grid.bus.values[0])
    energized = set(nx.node_connected_component(G, slack_bus)) if slack_bus in G else set()

    # Nodos: slack / energizado / no energizado
    slack_nodes = [slack_bus] if slack_bus in nodes else []
    energ_nodes = [n for n in nodes if (n in energized and n != slack_bus)]
    dead_nodes  = [n for n in nodes if n not in energized]

    # Dibujar nodos por capas (mejor control visual)
    ax.scatter(
        [pos[n][0] for n in energ_nodes],
        [pos[n][1] for n in energ_nodes],
        s=18, marker="o", edgecolors="black", linewidths=0.4,
        zorder=3
    )
    ax.scatter(
        [pos[n][0] for n in dead_nodes],
        [pos[n][1] for n in dead_nodes],
        s=18, marker="o", edgecolors="black", linewidths=0.4,
        zorder=3
    )
    ax.scatter(
        [pos[n][0] for n in slack_nodes],
        [pos[n][1] for n in slack_nodes],
        s=38, marker="s", edgecolors="black", linewidths=0.6,
        zorder=4
    )

    # Colores a posteriori (evitamos hardcode raro en scatter)
    # (Matplotlib no permite fácil mezcla por llamada, así que recoloreamos)
    # Simple: dibujar de nuevo con color específico encima
    ax.scatter([pos[n][0] for n in energ_nodes], [pos[n][1] for n in energ_nodes], s=18, marker="o",
            color="gray", zorder=3)
    ax.scatter([pos[n][0] for n in dead_nodes],  [pos[n][1] for n in dead_nodes],  s=18, marker="o",
            color="lightgray", zorder=3)
    ax.scatter([pos[n][0] for n in slack_nodes], [pos[n][1] for n in slack_nodes], s=38, marker="s",
            color="white", zorder=4)

    # Etiquetas (solo si no satura)
    # Si tienes muchos buses, esto puede ensuciar. Actívalo solo para demo.
    show_labels = False
    if show_labels:
        for n in nodes:
            x, y = pos[n]
            name = str(net.bus.at[n, "name"])
            ax.text(x, y + 0.03, name, fontsize=7, ha="center", va="bottom", zorder=6)

    # =======================================
    # 3) LÍNEAS NO switchables (base)
    # =======================================
    switchable_lines = set(net.switch[net.switch.et == "l"].element.values)

    for lid in net.line.index:
        if lid in switchable_lines:
            continue

        fb = int(net.line.at[lid, "from_bus"])
        tb = int(net.line.at[lid, "to_bus"])
        if fb not in pos_km or tb not in pos_km:
            continue

        x0, y0 = pos_km[fb]
        x1, y1 = pos_km[tb]

        ok = (self_line_status.get(lid, 1) == 1)
        ax.plot(
            [x0, x1], [y0, y1],
            color="black" if ok else "red",
            linewidth=0.9,
            alpha=0.9,
            zorder=1
        )

    # =======================================
    # 4) SWITCHES (paper convention)
    #   white = closed, black = open, red = failed line (override)
    #   Use double stroke so white is visible.
    # =======================================
    def draw_switch_segment(x0, y0, x1, y1, color, z=5):
        # base outline for visibility (esp. white)
        ax.plot([x0, x1], [y0, y1], color="gray", linewidth=2.2, linestyle="--", zorder=z-1)
        ax.plot([x0, x1], [y0, y1], color=color,  linewidth=1.4, linestyle="--", zorder=z)

    for sw_id, sw in net.switch.iterrows():
        x0 = y0 = x1 = y1 = None

        if sw.et == "l":
            line = int(sw.element)
            bus = int(sw.bus)

            fb = int(net.line.at[line, "from_bus"])
            tb = int(net.line.at[line, "to_bus"])
            other_bus = tb if bus == fb else fb

            x0, y0 = pos_km[bus]
            x1, y1 = pos_km[other_bus]

            line_ok = (self_line_status.get(line, 1) == 1)
            is_closed = bool(sw.closed)

            # priority: failure -> red
            if not line_ok:
                col = "red"
            elif is_closed:
                col = "white"
            else:
                col = "black"

            draw_switch_segment(x0, y0, x1, y1, col, z=5)

        elif sw.et == "b":
            b1 = int(sw.bus)
            b2 = int(sw.element)
            x0, y0 = pos_km[b1]
            x1, y1 = pos_km[b2]

            is_closed = bool(sw.closed)
            col = "white" if is_closed else "black"
            draw_switch_segment(x0, y0, x1, y1, col, z=5)

        # label switch id at midpoint (optional)
        # if x0 is not None:
        #     xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        #     ax.text(
        #         xm, ym, f"S{int(sw_id)}",
        #         fontsize=7, color="black",
        #         ha="center", va="center",
        #         zorder=6,
        #         bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6)
        #     )

    # =======================================
    # 5) TRANSFORMADORES
    # =======================================
    for tid, trafo in net.trafo.iterrows():
        hv = int(trafo.hv_bus)
        lv = int(trafo.lv_bus)
        if hv not in pos_km or lv not in pos_km:
            continue

        x0, y0 = pos_km[hv]
        x1, y1 = pos_km[lv]
        ax.plot(
            [x0, x1], [y0, y1],
            color="blue",
            linewidth=1.0,
            linestyle="-.",
            alpha=0.9,
            zorder=2
        )

    # =======================================
    # Leyenda (manual, paper-style)
    # =======================================
    legend_items = [
        Line2D([0],[0], marker="s", color="w", markerfacecolor="white",
            markeredgecolor="black", markersize=6, label="Slack bus"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
            markeredgecolor="black", markersize=5, label="Energized bus"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="lightgray",
            markeredgecolor="black", markersize=5, label="De-energized bus"),
        Line2D([0],[0], color="black", linewidth=1.0, label="Line in service"),
        Line2D([0],[0], color="red", linewidth=1.0, label="Line out of service"),
        Line2D([0],[0], color="black", linestyle="--", linewidth=1.4, label="Switch open"),
        Line2D([0],[0], color="white", linestyle="--", linewidth=1.4, label="Switch closed"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=True
    )

    # =======================================
    # Layout + save (PDF + PNG)
    # =======================================
    ax.set_title(f"CIGRE MV network state (t = {hour} h)")
    ax.set_xlabel("East–West distance [km]")
    ax.set_ylabel("North–South distance [km]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(f"figures/network_state_t{hour:02d}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/network_state_t{hour:02d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)