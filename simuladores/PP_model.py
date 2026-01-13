import mosaik_api
import pandapower as pp
import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import networkx as nx
from simuladores.logger import Logger
import json
import numpy as np

data_dir = './data/red_electrica'

def net_cigre15_mv(IN_PATH):
    """Carga la red CIGRE desde Excel exportado de pandapower."""
    CIG_MV_NET_PATH = os.path.join(IN_PATH, 'CIG_MV_net_rev1_23_09_25.xlsx')
    net = pp.from_excel(CIG_MV_NET_PATH)
    return net

def decode_geodata_string(s):
    """Convierte un string JSON escapado en dict con coordinates."""
    if not isinstance(s, str):
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "coordinates" in obj:
            return obj["coordinates"]
    except Exception:
        pass
    return None


META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'PPModel': {
            'public': True,
            'params': [],
            'attrs': [
                'line_status', # PPModel output
                'fail_prob', # FailureModel input
                'repair_plan', # OpDecisionModel input
                'climate', # ClimateModel input
                'ens', # PPModel output (evaluation)
                'line_positions',
                'grid_x',
                'grid_y',
                'shape',
                'lines',
                'buses',
                'switches',
                'transformers',
                'loads',
                'switch_plan',
            ],
        }
    },
}


class PPModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid = 'Grid'
        self.net = pp.create_empty_network()
        self.rng_sim = np.random.default_rng(2025)
        
        # Previous Config.
        self.line_status = {}
        
        # Inputs
        self.fail_prob = {}
        self.repair_plan = {}
        self.switch_plan = {}
        
        # Results
        self.logger = Logger("results/results.csv")
        self.lines = {}
        self.current_metrics = {}
        self.t = 0
        self.R_curve = {}
        self.switch_records = []
        
        # MonteCarlo
        self.M = 30            # nº muestras MC por step (empieza con 20-30)
        self.alpha = 0.90         # nivel para VaR/CVaR
        self.mc_stats = []        # guardar P90/CVaR por hora
        
        # Plot Net
        self.enable_plot = False # MUY importante en MC
        
        # -----------------------------
        # Inmunidad post-reparación
        # -----------------------------
        self.repair_immunity_s = 2 * 3600   # 2 horas (ajusta)
        self.immunity_until = {}           # {lid: time_s_hasta_el_que_es_inmune}

        
    # =============================================================
    # INITIALIZATION
    # =============================================================
        
    def init(self, sid, **sim_params):
        print("[PPModel] Inicializando simulador de red...")
        return META
    
    def create(self, num, model):
        """Crea entidad de red y configura topología."""
        print("📡 Recibido network_data desde mosaik_config.py")
        self.setup_network()

        self.current_metrics = {
            'ens': 0.0,
            'R_curve': self.R_curve
        }

        return [{'eid': self.eid, 'type': model, 'rel': []}]

    # ==============================================================
    # SETUP
    # ==============================================================

    def setup_network(self):
        """Carga la red CIGRE y convierte geodata escapado a coordenadas x,y."""
        
        print("📁 Cargando red CIGRE desde Excel...")
        IN_PATH = "./data/red_electrica"

        # ============================
        # 1) Cargar la red
        # ============================
        try:
            self.net = net_cigre15_mv(IN_PATH)
        except Exception as e:
            print(f"❌ Error cargando red CIGRE: {e}")
            raise e

        net = self.net
        print(f"🔌 Red cargada: {len(net.bus)} buses, {len(net.line)} líneas, {len(net.load)} cargas.")
        
        # ============================================
        # SWITCHES SEGÚN dnr_status
        # ============================================
        for lid in net.line.index:
            raw = str(net.line.at[lid, "dnr_status"]).lower()
            status = raw.replace("{", "").replace("}", "").strip()
            
            # ¿es switchable según el Excel?
            if status != "switchable":
                continue
            
            # Crear switch línea–bus en ambos extremos
            fb = net.line.at[lid, "from_bus"]
            tb = net.line.at[lid, "to_bus"]

            if lid not in net.switch.element.values:
                pp.create_switch(net, bus=fb, element=lid, et="l", closed=True)
                print(f"✔ Switches añadidos a línea switchable {lid}: {fb}-{tb}")
            
            

        # ============================
        # 2) Decodificar geodata de buses
        # ============================
        if "geo" in net.bus.columns:
            print("🛠 Decodificando geodata de buses...")
            for b in net.bus.index:
                coords = decode_geodata_string(net.bus.at[b, "geo"])
                if coords:
                    net.bus.at[b, "x"] = float(coords[0])*500
                    net.bus.at[b, "y"] = float(coords[1])*500

        # ============================
        # 4) Preparar lista de líneas y estados
        # ============================
        self.lines = dict(zip(net.line.index, net.line.name))
        self.line_status = {lid: 1 for lid in self.lines.keys()}

        print("✅ setup_network() completado.")

    

    # ==============================================================
    # SIMULATION STEP
    # ==============================================================

    def step(self, time, inputs, max_advance):
        print("\n==============================")
        self.t = int(time / 3600)
        print(f"⏱️  STEP t = {self.t} h")

        # Read inputs
        fail_prob = None
        repair_plan = None
        gust_speed = 0

        if inputs:
            src = list(inputs.keys())[0]
            vals = inputs[src]
            
            if 'fail_prob' in vals:
                fail_prob_all = list(vals['fail_prob'].values())[0]  # dict {k: {lid: q}}

                # 👉 usar SOLO el presente (k = 0)
                if isinstance(fail_prob_all, dict):
                    self.fail_prob = fail_prob_all.get(0, {})
                else:
                    self.fail_prob = {}
                
            if 'repair_plan' in vals:
                repair_plan = list(vals['repair_plan'].values())[0]
                self.repair_plan = repair_plan
                
            if 'switch_plan' in vals:
                switch_plan = list(vals['switch_plan'].values())[0]
                self.switch_plan = switch_plan
                
            for sid, state in self.switch_plan.items():
                self.net.switch.at[int(sid), "closed"] = bool(state)

            for lid, data in self.repair_plan.items():
                if data.get('finish_time',0) <= self.t * 3600:
                    self.line_status[lid] = 1
                    self.net.line.at[int(lid), "in_service"] = True
                    
                    # Inmunidad
                    self.immunity_until[lid] = time + self.repair_immunity_s
                    
            self.immunity_until = {lid: until for lid, until in self.immunity_until.items() if until > time}
                    

            if 'climate' in vals:
                gust_speed = list(vals["climate"].values())[0][0]["gust_speed"]
                self.last_gust_speed = np.array(gust_speed)

            if 'grid_x' in vals:
                self.grid_x = np.array(list(vals['grid_x'].values())[0])
            if 'grid_y' in vals:
                self.grid_y = np.array(list(vals['grid_y'].values())[0])
            if 'shape' in vals:
                self.shape = tuple(list(vals['shape'].values())[0])


        t = time / 3600.0
        
        # MonteCarlo Simulation
        expected_load = float(self.net.load.p_mw.sum())
        line_ids = list(self.net.line.index)        
        
        # Guardar estado base del step (antes de muestrear)
        base_in_service = self.net.line["in_service"].copy()
        base_status = self.line_status.copy()
        
        ens_samples = np.empty(self.M, dtype=float)
        cache_pf = {}  # key(bytes) -> ens_mw
        
        for m in range(self.M):
            # Reset al estado base del step
            self.net.line["in_service"] = base_in_service
            self.line_status = base_status.copy()

            for i, lid in enumerate(line_ids):
                lid = int(lid)

                # si ya estaba caída, permanece caída
                if base_status.get(lid, 1) == 0:
                    self.line_status[lid] = 0
                    continue
                
                # Inmunidad
                if self.immunity_until.get(lid, 0) > time:
                    self.line_status[lid] = 1
                    continue

                p = float(self.fail_prob.get(lid, 0.0))
                self.line_status[lid] = 0 if (self.rng_sim.random() < p) else 1

            # clave compacta: bitstring -> bytes
            state_vec = np.array(
                [self.line_status.get(lid, 1) for lid in line_ids],
                dtype=np.uint8
            )
            key = np.packbits(state_vec).tobytes()

            # ===== cache hit =====
            if key in cache_pf:
                ens_mw = cache_pf[key]
                ens_samples[m] = ens_mw
                served_load = expected_load - ens_mw
                
                for i, lid in enumerate(line_ids):
                    self.net.line.at[int(lid), "in_service"] = bool(self.line_status[lid])
                
                continue

            # ===== aplicar estados sampleados =====
            for i, lid in enumerate(line_ids):
                self.net.line.at[int(lid), "in_service"] = bool(self.line_status[lid])

            # ===== PF =====
            try:
                pp.runpp(self.net)
                served_load = float(self.net.res_load.p_mw.sum())
                ens_mw = max(0.0, expected_load - served_load)
            except Exception:
                ens_mw = expected_load

            # guardar en cache y en muestras
            cache_pf[key] = ens_mw
            ens_samples[m] = ens_mw
            
        # Actualizar líneas según el modelo de fallo
        for lid, status in self.line_status.items():
            self.net.line.at[lid, 'in_service'] = bool(status)
            
        for sw, state in self.switch_plan.items(): 
                line_id = self.net.switch.at[sw, "element"] 
                sw_in_service = bool(self.net.line.at[line_id, "in_service"]) 
                self.switch_records.append({ "time": t, "switch": sw, "state": state, "sw_in_service": sw_in_service })

        # VaR (P90) y CVaR0.9
        p90 = float(np.quantile(ens_samples, self.alpha))
        tail = ens_samples[ens_samples >= p90]
        cvar90 = float(tail.mean()) if tail.size else p90
        
        # R-Curve CVaR90
        R = (expected_load-cvar90)/expected_load
        # R = served_load/expected_load
        self.R_curve[self.t] = R
        ens = max(0.0, expected_load - served_load)   

        # Guardar resultados y plot
        if self.enable_plot:
            self.plot_network(self.t)
        
        df_switches = pd.DataFrame(self.switch_records)
        df_switches.to_csv("results/switch_states_GA_PRUEBA.csv", index=False)
        
        if time >= (24 * 3600) - 3600:
            # último paso de simulación
            self.plot_R_curve()

        # Preparar salida
        self.current_metrics = {
            'ens': ens,
            'R_curve': self.R_curve
        }
        
        self.mc_stats.append({
            "t": self.t,
            "ens_mean": float(ens_samples.mean()),
            "ens_p50": float(np.quantile(ens_samples, 0.50)),
            "ens_p90": p90,
            "ens_cvar90": cvar90,
            "ens_max": float(ens_samples.max()),
            "ens_chosen": float(ens),
        })
        
        print(f"[MC-DC] ENS mean={ens_samples.mean():.3f} MW | P90={p90:.3f} MW | CVaR0.9={cvar90:.3f} MW | chosen={ens:.3f} MW")

        return time + 3600

    def plot_R_curve(self):
        import os
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        R = self.current_metrics.get("R_curve", {})
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


    # ==============================================================
    # UTILIDADES
    # ==============================================================

    def save_results(self, hour, wind_speed, ens):
        filename = f"results/hour_{hour:02d}.csv"
        fieldnames = ["hour", "wind_speed", "ens"] + [str(l) for l in self.lines.keys()]

        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            row = {"hour": hour, "wind_speed": wind_speed, "ens": ens}
            row.update({str(l): self.line_status.get(l, 1) for l in self.lines.keys()})
            writer.writerow(row)

    def get_data(self, outputs=None):
        """Devuelve ENS, corrientes y posiciones de línea a mosaik."""
        line_pos = {}
        for i, row in self.net.line.iterrows():
            bus0 = self.net.bus.at[row.from_bus, 'name']
            bus1 = self.net.bus.at[row.to_bus, 'name']
            # Leer geodata de forma segura. Pandapower guarda la tupla en la columna 'geodata'.
            def _get_geodata(idx):
                """
                Transforms data from pp dictionary and makes it plottable
                From str to float

                Args:
                    idx: id of the bus

                Returns:
                    tuple containing position of the bus in float
                """
                if 'x' in self.net.bus.columns and 'y' in self.net.bus.columns:
                    try:
                        return (float(self.net.bus.at[idx, 'x']), float(self.net.bus.at[idx, 'y']))
                    except Exception:
                        return (0.0, 0.0)
                return (0.0, 0.0)

            x0, y0 = _get_geodata(row.from_bus)
            x1, y1 = _get_geodata(row.to_bus)
            line_pos[row.name] = {"bus0": bus0, "bus1": bus1, "x0": x0, "y0": y0, "x1": x1, "y1": y1}

        return {
            self.eid: {
                'ens': self.current_metrics.get('ens', 0.0),
                'line_positions': line_pos,
                'line_status': self.line_status,
                'lines': self.net.line,
                'buses': self.net.bus,
                'switches': self.net.switch,
                'transformers': self.net.trafo,
                'loads': self.net.load,
            }
        }

    def plot_network(self, hour):
        import os
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import networkx as nx
        import pandapower.topology as top
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

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

        net = self.net
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
        if hasattr(self, "last_gust_speed") and isinstance(self.last_gust_speed, np.ndarray):
            if hasattr(self, "grid_x") and hasattr(self, "grid_y"):
                X, Y = np.meshgrid(self.grid_x, self.grid_y)

                im = ax.imshow(
                    self.last_gust_speed,
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

            ok = (self.line_status.get(lid, 1) == 1)
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

                line_ok = (self.line_status.get(line, 1) == 1)
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

    # def plot_network(self, hour):
    #     import matplotlib.pyplot as plt
    #     import networkx as nx
    #     import numpy as np
    #     import pandapower.topology as top
    #     import os

    #     plt.figure(figsize=(8, 6))
    #     net = self.net

    #     # =======================================
    #     #  GRAFO REAL (respeta switches)
    #     # =======================================
    #     G = top.create_nxgraph(net, respect_switches=True, include_out_of_service=False)

    #     # =======================================
    #     # POSICIONES (convertir a km y centrar)
    #     # =======================================
    #     FT_TO_KM = 0.0003048
    #     pos_km = {
    #         bus: (float(net.bus.at[bus, "x"]) * FT_TO_KM,
    #             float(net.bus.at[bus, "y"]) * FT_TO_KM)
    #         for bus in net.bus.index
    #     }

    #     # Centrado
    #     xs = [p[0] for p in pos_km.values()]
    #     ys = [p[1] for p in pos_km.values()]
    #     x_mean, y_mean = np.mean(xs), np.mean(ys)
    #     pos_km = {b: (pos_km[b][0] - x_mean, pos_km[b][1] - y_mean) for b in pos_km}

    #     # =======================================
    #     # 1) DIBUJAR VIENTO
    #     # =======================================
    #     if hasattr(self, "last_gust_speed") and isinstance(self.last_gust_speed, np.ndarray):
    #         if hasattr(self, "grid_x") and hasattr(self, "grid_y"):
    #             X, Y = np.meshgrid(self.grid_x, self.grid_y)
    #             plt.imshow(
    #                 self.last_gust_speed,
    #                 origin="lower",
    #                 cmap="coolwarm",
    #                 alpha=0.5,
    #                 extent=[X.min(), X.max(), Y.min(), Y.max()],
    #                 vmin=10, vmax=30
    #             )
    #             plt.colorbar(label="Velocidad del viento [m/s]", shrink=0.7)

    #     # =======================================
    #     # 2) BUSES ENERGIZADOS (topología real)
    #     # =======================================
    #     slack_bus = int(net.ext_grid.bus.values[0])
    #     energized = set(nx.node_connected_component(G, slack_bus)) if slack_bus in G else set()

    #     nodes = list(G.nodes())
    #     pos_filtered = {n: pos_km[n] for n in nodes}

    #     node_colors = []
    #     node_sizes = []

    #     for b in nodes:
    #         if b == slack_bus:
    #             node_colors.append("limegreen")
    #             node_sizes.append(20)
    #         elif b in energized:
    #             node_colors.append("green")
    #             node_sizes.append(10)
    #         else:
    #             node_colors.append("red")
    #             node_sizes.append(10)

    #     nx.draw_networkx_nodes(
    #         G,
    #         pos_filtered,
    #         nodelist=nodes,
    #         node_color=node_colors,
    #         node_size=node_sizes,
    #         edgecolors="black"
    #     )
        
    #     # =======================================
    #     # DIBUJAR NOMBRES DE LOS NODOS
    #     # =======================================
    #     for bus_idx in nodes:
    #         x, y = pos_filtered[bus_idx]
    #         label = str(net.bus.at[bus_idx, "name"])
            
    #         plt.text(
    #             x, y + 0.03,          # pequeño desplazamiento vertical
    #             label,
    #             fontsize=7,
    #             ha="center",
    #             va="bottom"
    #         )

    #     # =======================================
    #     # 3) DIBUJAR LÍNEAS NO SWITCHABLES
    #     # =======================================
    #     switchable_lines = set(net.switch[net.switch.et == "l"].element.values)

    #     for lid in net.line.index:
    #         if lid in switchable_lines:
    #             continue  # ❌ NO dibujar líneas conmutables

    #         fb = net.line.at[lid, "from_bus"]
    #         tb = net.line.at[lid, "to_bus"]

    #         if fb in pos_km and tb in pos_km:
    #             x0, y0 = pos_km[fb]
    #             x1, y1 = pos_km[tb]

    #             plt.plot(
    #                 [x0, x1], [y0, y1],
    #                 color="green" if self.line_status.get(lid, 1) == 1 else "red",
    #                 linestyle="-",
    #                 linewidth=0.8,
    #                 alpha=0.8,
    #             )

    #     # =======================================
    #     # 4) DIBUJAR SWITCHES
    #     # =======================================
    #     for sw_id, sw in net.switch.iterrows():

    #         # ---- bus-line switch ----
    #         if sw.et == "l":
    #             line = sw.element
    #             bus = sw.bus

    #             fb = net.line.at[line, "from_bus"]
    #             tb = net.line.at[line, "to_bus"]

    #             other_bus = tb if bus == fb else fb

    #             x0, y0 = pos_km[bus]
    #             x1, y1 = pos_km[other_bus]

    #             plt.plot(
    #                 [x0, x1], [y0, y1],
    #                 color="green" if sw.closed and self.line_status.get(line, 1) else "red",
    #                 linewidth=1.3,
    #                 linestyle="--",
    #                 zorder=5
    #             )

    #         # ---- bus-bus switch ----
    #         elif sw.et == "b":
    #             b1 = sw.bus
    #             b2 = sw.element

    #             x0, y0 = pos_km[b1]
    #             x1, y1 = pos_km[b2]

    #             plt.plot(
    #                 [x0, x1], [y0, y1],
    #                 color="green" if sw.closed else "red",
    #                 linewidth=1.3,
    #                 linestyle="--",
    #                 zorder=5
    #             )
                
    #         # ---- label en el punto medio ----
    #         xm = 0.5 * (x0 + x1)
    #         ym = 0.5 * (y0 + y1)

    #         plt.text(
    #             xm, ym,
    #             f"S{sw_id}",
    #             fontsize=7,
    #             color="black",
    #             ha="center",
    #             va="center",
    #             zorder=6,
    #             bbox=dict(
    #                 facecolor="white",
    #                 edgecolor="none",
    #                 alpha=0.7,
    #                 pad=0.8
    #             )
    #         )
                
    #     # =======================================
    #     # 5) DIBUJAR TRANSFORMADORES (hv_bus ↔ lv_bus)
    #     # =======================================

    #     for tid, trafo in net.trafo.iterrows():
    #         hv = trafo.hv_bus
    #         lv = trafo.lv_bus

    #         if hv in pos_km and lv in pos_km:
    #             x0, y0 = pos_km[hv]
    #             x1, y1 = pos_km[lv]

    #             plt.plot(
    #                 [x0, x1], [y0, y1],
    #                 color="blue",
    #                 linewidth=1.0,
    #                 linestyle="--",
    #                 alpha=0.9,
    #                 zorder=3,
    #             )

    #             # Opcional: etiqueta del trafo
    #             mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    #             plt.text(mx, my + 0.03, f"T{tid}", color="blue", fontsize=7, ha="center")

    #     # =======================================
    #     # 6) FORMATO + GUARDADO
    #     # =======================================
    #     plt.title(f"Red CIGRE - Topología y viento (t = {hour} h)")
    #     plt.xlabel("Distancia Este–Oeste [km]")
    #     plt.ylabel("Distancia Norte–Sur [km]")
    #     plt.gca().set_aspect("equal", adjustable="box")
    #     plt.grid(alpha=0.3)
    #     plt.tight_layout()

    #     os.makedirs("figures", exist_ok=True)
    #     plt.savefig(f"figures/hour_{hour:02d}.png", dpi=200, bbox_inches="tight")
    #     plt.close()
