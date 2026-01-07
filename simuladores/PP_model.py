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

np.random.seed(2025)   # cualquier número fijo

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
        self.line_status = {}
        self.fail_prob = {}
        self.repair_plan = {}
        self.switch_plan = {}
        self.logger = Logger("results/results.csv")
        self.lines = {}
        self.current_metrics = {}
        self.t = 0
        self.R_curve = {}
        self.switch_records = []
        
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
            
            for lid, prob in self.fail_prob.items():
                if self.line_status.get(lid,1) == 1 and np.random.rand() < prob:
                    self.line_status[lid] = 0
                    
            for lid, data in self.repair_plan.items():
                if data.get('finish_time',0) <= self.t * 3600:
                    self.line_status[lid] = 1

            if 'climate' in vals:
                gust_speed = list(vals["climate"].values())[0][0]["gust_speed"]
                self.last_gust_speed = np.array(gust_speed)

            if 'grid_x' in vals:
                self.grid_x = np.array(list(vals['grid_x'].values())[0])
            if 'grid_y' in vals:
                self.grid_y = np.array(list(vals['grid_y'].values())[0])
            if 'shape' in vals:
                self.shape = tuple(list(vals['shape'].values())[0])

        # Actualizar líneas según el modelo de fallo
        for lid, status in self.line_status.items():
            self.net.line.at[lid, 'in_service'] = bool(status)

        t = time / 3600.0

        for sw, state in self.switch_plan.items():
            line_id = self.net.switch.at[sw, "element"]
            sw_in_service = bool(self.net.line.at[line_id, "in_service"])
            self.switch_records.append({
                "time": t,
                "switch": sw,
                "state": state,
                "sw_in_service": sw_in_service
            })

        df_switches = pd.DataFrame(self.switch_records)
        df_switches.to_csv("results/switch_states_GA_PRUEBA.csv", index=False)
        
        # Calcular flujo DC
        try:
            pp.runpp(self.net)
            # pp.rundcpp(self.net)
            print("DC power flow calculado correctamente.")
            
            # Calcular ENS topológica
            expected_load = self.net.load.p_mw.sum()
            served_load = self.net.res_load.p_mw.sum()
            R = served_load/expected_load
            self.R_curve[self.t] = R
            
            ens = max(0.0, expected_load - served_load)
            print(f"Expected: {expected_load:.3f} MW, Served: {served_load:.3f} MW, ENS: {ens:.3f} MW")
            
        except Exception as e:
            ens = self.net.load.p_mw.sum()
            print(f"Error en rundcpp: {e}")
            return time + 3600

        # Guardar resultados y plot
        # os.makedirs("results", exist_ok=True)
        # self.save_results(self.t, wind_speed, ens)
        self.plot_network(self.t)
        
        if time >= (24 * 3600) - 3600:
            # último paso de simulación
            self.plot_R_curve()

        # Preparar salida
        self.current_metrics = {
            'ens': ens,
            'R_curve': self.R_curve
        }

        return time + 3600
    
    def plot_R_curve(self):
        import matplotlib.pyplot as plt, os
        R = self.current_metrics.get("R_curve", {})
        if not R:
            return
        times = list(R.keys())
        values = [float(v) for v in R.values()]
        os.makedirs("results", exist_ok=True)
        plt.plot(times, values, marker="o")
        plt.title("Curva de Resiliencia R(t)")
        plt.xlabel("Hora")
        plt.ylabel("R(t)")
        plt.grid(alpha=0.4)
        plt.savefig("results/R_curve.png", dpi=200)
        plt.close()
        print("Guardada R_curve en results/R_curve.png")

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
            }
        }

    def plot_network(self, hour):
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
        import pandapower.topology as top
        import os

        plt.figure(figsize=(8, 6))
        net = self.net

        # =======================================
        #  GRAFO REAL (respeta switches)
        # =======================================
        G = top.create_nxgraph(net, respect_switches=True, include_out_of_service=False)

        # =======================================
        # POSICIONES (convertir a km y centrar)
        # =======================================
        FT_TO_KM = 0.0003048
        pos_km = {
            bus: (float(net.bus.at[bus, "x"]) * FT_TO_KM,
                float(net.bus.at[bus, "y"]) * FT_TO_KM)
            for bus in net.bus.index
        }

        # Centrado
        xs = [p[0] for p in pos_km.values()]
        ys = [p[1] for p in pos_km.values()]
        x_mean, y_mean = np.mean(xs), np.mean(ys)
        pos_km = {b: (pos_km[b][0] - x_mean, pos_km[b][1] - y_mean) for b in pos_km}

        # =======================================
        # 1) DIBUJAR VIENTO
        # =======================================
        if hasattr(self, "last_gust_speed") and isinstance(self.last_gust_speed, np.ndarray):
            if hasattr(self, "grid_x") and hasattr(self, "grid_y"):
                X, Y = np.meshgrid(self.grid_x, self.grid_y)
                plt.imshow(
                    self.last_gust_speed,
                    origin="lower",
                    cmap="coolwarm",
                    alpha=0.5,
                    extent=[X.min(), X.max(), Y.min(), Y.max()],
                    vmin=0, vmax=30
                )
                plt.colorbar(label="Velocidad del viento [m/s]", shrink=0.7)

        # =======================================
        # 2) BUSES ENERGIZADOS (topología real)
        # =======================================
        slack_bus = int(net.ext_grid.bus.values[0])
        energized = set(nx.node_connected_component(G, slack_bus)) if slack_bus in G else set()

        nodes = list(G.nodes())
        pos_filtered = {n: pos_km[n] for n in nodes}

        node_colors = []
        node_sizes = []

        for b in nodes:
            if b == slack_bus:
                node_colors.append("limegreen")
                node_sizes.append(20)
            elif b in energized:
                node_colors.append("green")
                node_sizes.append(10)
            else:
                node_colors.append("red")
                node_sizes.append(10)

        nx.draw_networkx_nodes(
            G,
            pos_filtered,
            nodelist=nodes,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="black"
        )
        
        # =======================================
        # DIBUJAR NOMBRES DE LOS NODOS
        # =======================================
        for bus_idx in nodes:
            x, y = pos_filtered[bus_idx]
            label = str(net.bus.at[bus_idx, "name"])
            
            plt.text(
                x, y + 0.03,          # pequeño desplazamiento vertical
                label,
                fontsize=7,
                ha="center",
                va="bottom"
            )

        # =======================================
        # 3) DIBUJAR LÍNEAS NO SWITCHABLES
        # =======================================
        switchable_lines = set(net.switch[net.switch.et == "l"].element.values)

        for lid in net.line.index:
            if lid in switchable_lines:
                continue  # ❌ NO dibujar líneas conmutables

            fb = net.line.at[lid, "from_bus"]
            tb = net.line.at[lid, "to_bus"]

            if fb in pos_km and tb in pos_km:
                x0, y0 = pos_km[fb]
                x1, y1 = pos_km[tb]

                plt.plot(
                    [x0, x1], [y0, y1],
                    color="green" if self.line_status.get(lid, 1) == 1 else "red",
                    linestyle="-",
                    linewidth=0.8,
                    alpha=0.8,
                )

        # =======================================
        # 4) DIBUJAR SWITCHES
        # =======================================
        for sw_id, sw in net.switch.iterrows():

            # ---- bus-line switch ----
            if sw.et == "l":
                line = sw.element
                bus = sw.bus

                fb = net.line.at[line, "from_bus"]
                tb = net.line.at[line, "to_bus"]

                other_bus = tb if bus == fb else fb

                x0, y0 = pos_km[bus]
                x1, y1 = pos_km[other_bus]

                plt.plot(
                    [x0, x1], [y0, y1],
                    color="green" if sw.closed and self.line_status.get(line, 1) else "red",
                    linewidth=1.3,
                    linestyle="--",
                    zorder=5
                )

            # ---- bus-bus switch ----
            elif sw.et == "b":
                b1 = sw.bus
                b2 = sw.element

                x0, y0 = pos_km[b1]
                x1, y1 = pos_km[b2]

                plt.plot(
                    [x0, x1], [y0, y1],
                    color="green" if sw.closed else "red",
                    linewidth=1.3,
                    linestyle="--",
                    zorder=5
                )
                
            # ---- label en el punto medio ----
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)

            plt.text(
                xm, ym,
                f"S{sw_id}",
                fontsize=7,
                color="black",
                ha="center",
                va="center",
                zorder=6,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    pad=0.8
                )
            )
                
        # =======================================
        # 5) DIBUJAR TRANSFORMADORES (hv_bus ↔ lv_bus)
        # =======================================

        for tid, trafo in net.trafo.iterrows():
            hv = trafo.hv_bus
            lv = trafo.lv_bus

            if hv in pos_km and lv in pos_km:
                x0, y0 = pos_km[hv]
                x1, y1 = pos_km[lv]

                plt.plot(
                    [x0, x1], [y0, y1],
                    color="blue",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.9,
                    zorder=3,
                )

                # Opcional: etiqueta del trafo
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                plt.text(mx, my + 0.03, f"T{tid}", color="blue", fontsize=7, ha="center")

        # =======================================
        # 6) FORMATO + GUARDADO
        # =======================================
        plt.title(f"Red CIGRE - Topología y viento (t = {hour} h)")
        plt.xlabel("Distancia Este–Oeste [km]")
        plt.ylabel("Distancia Norte–Sur [km]")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/hour_{hour:02d}.png", dpi=200, bbox_inches="tight")
        plt.close()
