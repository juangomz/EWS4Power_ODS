import mosaik_api
import pandapower as pp
import pandapower.topology as top
import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import networkx as nx
from simuladores.logger import Logger
import json
import pandapower.converter as pc

np.random.seed(42)   # cualquier número fijo

data_dir = './data/red_electrica'

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
                'gust_speed', # ClimateModel input
                'ens', # PPModel output (evaluation)
                'line_positions',
                'grid_x',
                'grid_y',
                'shape',
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
        self.logger = Logger("results/results.csv")
        self.lines = {}
        self.current_metrics = {}
        self.t = 0
        self.R_curve = {}
        
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
        """Carga la red eléctrica desde los CSVs y crea el modelo pandapower."""
        
        data_dir = "feeder123"
        print("⚙️  Cargando datos del IEEE123...")

        # === Leer archivos ===
        lines_xls = pd.read_excel(
            os.path.join(data_dir, "line data.xls"),
            skiprows=2
        )
        lines_xls['Node A'] = lines_xls['Node A'].astype(str).str.strip()
        lines_xls['Node B'] = lines_xls['Node B'].astype(str).str.strip()
        config_xls = pd.read_excel(os.path.join(data_dir, "config data.xls"),
                                   skiprows=2
        )
        loads_xls = pd.read_excel(os.path.join(data_dir, "spot loads data.xls"),
                                  skiprows=2
        )
        loads_xls['Node']   = loads_xls['Node'].astype(str).str.strip()

        coords = pd.read_csv("feeder123/BusCoords.csv", header=None, sep=r"[;\s]+", engine="python", decimal=',')
        coords.columns = ["bus", "x", "y"]
        coords["bus"] = coords["bus"].astype(str).str.strip()

        for name in coords["bus"]:
            pp.create_bus(self.net, vn_kv=4.16, name=str(name))

        for _, r in coords.iterrows():
            m = self.net.bus[self.net.bus.name == r["bus"]]
            if not m.empty:
                idx = m.index[0]
                self.net.bus.at[idx, "x"] = float(r["x"])
                self.net.bus.at[idx, "y"] = float(r["y"])

        # === Crear líneas ===
        for _, row in lines_xls.iterrows():
            try:
                bus0 = int(row['Node A'].strip())
                bus1 = int(row['Node B'].strip())
                
                busA = self.net.bus[self.net.bus["name"] == str(bus0)].index[0]
                busB = self.net.bus[self.net.bus["name"] == str(bus1)].index[0]
                
                length_km = row['Length (ft.)'] * 0.0003048 # pasar de pies a km

                cfg = str(row['Config.']).strip()
                cfg_map = {
                    1: (0.4, 0.05),
                    2: (0.5, 0.06),
                    3: (0.6, 0.08),
                    4: (0.4, 0.05),
                    5: (0.5, 0.06),
                    6: (0.5, 0.07),
                    7: (0.4, 0.05),
                    8: (0.5, 0.06),
                    9: (0.4, 0.05),
                    10: (0.5, 0.06),
                    11: (0.4, 0.05),
                    12: (0.4, 0.05),
                }

                r_ohm_per_km, x_ohm_per_km = cfg_map.get(cfg, (0.5, 0.06))

                name = f"Line_{row['Node A']}_{row['Node B']}"

                pp.create_line_from_parameters(
                    self.net,
                    from_bus=busA,     # ← SIEMPRE Node A
                    to_bus=busB,       # ← SIEMPRE Node B
                    length_km=length_km,
                    r_ohm_per_km=r_ohm_per_km,
                    x_ohm_per_km=x_ohm_per_km,
                    c_nf_per_km=0.0,
                    max_i_ka=0.2,
                    name=name
                )
            except Exception as e:
                print(f"⚠️  Error creando línea {row.get('Node A', '?')}-{row.get('Node B', '?')}: {e}")
                
        # Punetes entre ramas desconectadas
        bridge_pairs = [
            ("18", "135"),
            ("13", "152"),
            ("54", "94"),
            ("151", "300"),
            ("97", "197"),
            ("60", "160"),
        ]

        # Incluir puentes en la red
        for a, b in bridge_pairs:
            if a in self.net.bus.name.values and b in self.net.bus.name.values:
                bus_a = self.net.bus[self.net.bus.name == a].index[0]
                bus_b = self.net.bus[self.net.bus.name == b].index[0]
                if int(a) in (13,18, 54, 151):
                    closed = True
                else:
                    closed = False
                pp.create_switch(
                    self.net,
                    bus=bus_a,
                    element=bus_b,
                    et="b",
                    closed=closed
                )
                print(f"🔗 Puente añadido entre {a} y {b}")

        # === Crear cargas ===
        for _, row in loads_xls.iterrows():
            try:
                bus_name = str(row['Node']).strip()
                if bus_name not in self.net.bus.name.values:
                    continue
                bus_idx = self.net.bus[self.net.bus.name == bus_name].index[0]
                # Sumar fases si hay más de una
                p_kw = sum([v for k, v in row.items() if 'Ph-' in k and isinstance(v, (int, float)) and 'kW' not in k])
                pp.create_load(self.net, bus=bus_idx, p_mw=p_kw / 1000, name=f"Load_{bus_name}")
            except Exception as e:
                print(f"⚠️  Error creando carga en {bus_name}: {e}")
                
        if "150" not in self.net.bus["name"].values:
            b150 = pp.create_bus(self.net, vn_kv=4.16, name="150")
            self.net.bus.at[b150, "x"] = 100
            self.net.bus.at[b150, "y"] = 1500

        # 2. Índices de buses por nombre (OJO: strings)
        bus150_idx = self.net.bus[self.net.bus.name == "150"].index[0]
        bus149_idx = self.net.bus[self.net.bus.name == "149"].index[0]

        # 3. Switch bus–bus entre 150 y 149
        pp.create_switch(
            self.net,
            bus=bus150_idx,
            element=bus149_idx,
            et="b",      # bus-bus switch
            closed=True
        )
        
        # === Crear barra slack ===
        pp.create_ext_grid(self.net, bus=bus150_idx, vm_pu=1.0)
        
        self.lines = dict(zip(self.net.line.index, self.net.line.name))
        self.line_status = {lid: 1 for lid in self.lines.keys()}

        print(f"✅ Red creada: {len(self.net.bus)} buses, {len(self.net.line)} líneas, {len(self.net.load)} cargas.")
    

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
                fail_prob = list(vals['fail_prob'].values())[0]
                self.fail_prob = fail_prob
                
            if 'repair_plan' in vals:
                repair_plan = list(vals['repair_plan'].values())[0]
                self.repair_plan = repair_plan
            
            for lid, prob in self.fail_prob.items():
                if self.line_status.get(lid,1) == 1 and np.random.rand() < prob:
                    self.line_status[lid] = 0
                    
            for lid, data in self.repair_plan.items():
                if data.get('finish_time',0) <= self.t * 3600:
                    self.line_status[lid] = 1

            if 'gust_speed' in vals:
                gust_speed = list(vals['gust_speed'].values())[0]
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

        # Calcular flujo DC
        try:
            pp.runpp(self.net)
            # pp.rundcpp(self.net)
            print("✅ DC power flow calculado correctamente.")
            
            # Calcular ENS topológica
            expected_load = self.net.load.p_mw.sum()
            served_load = self.net.res_load.p_mw.sum()
            R = served_load/expected_load
            self.R_curve[self.t] = R
            
            ens = max(0.0, expected_load - served_load)
            print(f"Expected: {expected_load:.3f} MW, Served: {served_load:.3f} MW, ENS: {ens:.3f} MW")
            
        except Exception as e:
            ens = self.net.load.p_mw.sum()
            print(f"⚠️ Error en rundcpp: {e}")
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
        print("📈 Guardada R_curve en results/R_curve.png")

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
                # 'num_lines': len(self.net.line),
                'line_positions': line_pos,
                'line_status': self.line_status
            }
        }

    # def plot_network(self, hour):
    #     import matplotlib.pyplot as plt
    #     import networkx as nx
    #     import numpy as np
    #     import pandapower.topology as top
    #     import os

    #     plt.figure(figsize=(8, 6))
    #     net = self.net

    #     # Crear grafo
    #     G = top.create_nxgraph(net)

    #     # Conversión de pies → kilómetros
    #     FT_TO_KM = 0.0003048
    #     pos_km = {
    #         bus: (
    #             float(net.bus.at[bus, "x"]) * FT_TO_KM,
    #             float(net.bus.at[bus, "y"]) * FT_TO_KM,
    #         )
    #         for bus in net.bus.index
    #     }

    #     # Centrar figura (coincide con el dominio del viento)
    #     x_vals = [v[0] for v in pos_km.values()]
    #     y_vals = [v[1] for v in pos_km.values()]
    #     x_mean, y_mean = np.mean(x_vals), np.mean(y_vals)
    #     pos_km = {b: (x - x_mean, y - y_mean) for b, (x, y) in pos_km.items()}

    #     # ==========================
    #     # PLOT DEL VIENTO
    #     # ==========================
    #     if hasattr(self, "last_gust_speed") and isinstance(self.last_gust_speed, np.ndarray):
    #         if hasattr(self, "grid_x") and hasattr(self, "grid_y"):
    #             X, Y = np.meshgrid(self.grid_x, self.grid_y)
    #         elif hasattr(self, "wind_grid_lon") and hasattr(self, "wind_grid_lat"):
    #             X, Y = np.meshgrid(self.wind_grid_lon, self.wind_grid_lat)
    #         else:
    #             X, Y = None, None

    #         if X is not None and Y is not None:
    #             extent = [X.min(), X.max(), Y.min(), Y.max()]
    #             plt.imshow(
    #                 self.last_gust_speed,
    #                 origin="lower",
    #                 cmap="coolwarm",
    #                 alpha=0.5,
    #                 extent=extent,
    #                 vmin=0,
    #                 vmax=30,
    #             )
    #             plt.colorbar(label="Velocidad del viento [m/s]", shrink=0.7)
    #         else:
    #             print("⚠️ No hay malla de viento (X, Y) para graficar.")
    #     else:
    #         print("⚠️ No hay campo de viento disponible para graficar.")

    #     # ==========================
    #     # 💡 COLOREAR LOS BUSES SEGÚN SI ESTÁN ENERGIZADOS
    #     # ==========================
    #     slack_bus_idx = int(net.ext_grid.bus.values[0]) if len(net.ext_grid.bus) > 0 else None

    #     node_colors = []
    #     node_sizes = []
    #     for bus in net.bus.index:
    #         if bus in net.res_bus.index:
    #             vm = net.res_bus.at[bus, "vm_pu"]
    #             if np.isnan(vm) or vm < 0.1:
    #                 color = "red"
    #             else:
    #                 color = "skyblue"
    #         else:
    #             color = "red"

    #         # 💡 Marcar el bus generador (slack)
    #         if bus == slack_bus_idx:
    #             color = "limegreen"  # verde brillante
    #             size = 80            # más grande
    #         else:
    #             size = 20

    #         node_colors.append(color)
    #         node_sizes.append(size)

    #     # # Si usas nombres en lugar de índices:
    #     # labels = {bus: str(net.bus.at[bus, "name"]) for bus in net.bus.index}
        
    #     # pos = pos_km  # o el nombre que uses para tu dict de posiciones
        
    #     # # Creamos las posiciones desplazadas para las etiquetas
    #     # label_pos = {node: (pos[node][0], pos[node][1] + 0.05) for node in pos}

    #     # nx.draw_networkx_labels(
    #     #     G,
    #     #     label_pos,
    #     #     labels=labels,
    #     #     font_size=3,         # tamaño pequeño para no saturar
    #     #     font_color="black",  # o "darkgreen" si prefieres sobre el fondo
    #     #     verticalalignment="center",
    #     #     horizontalalignment="center"
    #     # )
        
    #     # ==========================
    #     # PLOT DE LA RED
    #     # ==========================
    #     nx.draw_networkx_nodes(G, pos_km, node_color=node_colors, node_size=8, edgecolors="black")

    #     for lid in net.line.index:
    #         bus0 = net.line.at[lid, "from_bus"]
    #         bus1 = net.line.at[lid, "to_bus"]
    #         if bus0 in pos_km and bus1 in pos_km:
    #             x0, y0 = pos_km[bus0]
    #             x1, y1 = pos_km[bus1]
    #             plt.plot(
    #                 [x0, x1], [y0, y1],
    #                 color="green" if self.line_status.get(lid, 1) == 1 else "red",
    #                 linewidth=0.8,
    #                 linestyle="--" if self.line_status.get(lid, 1) == 0 else "-",
    #                 alpha=0.8,
    #             )

    #     # ============================
    #     # DIBUJAR LOS SWITCHES
    #     # ============================

    #     for sw_idx, sw in net.switch.iterrows():

    #         bus_a = sw.bus
    #         closed = sw.closed
    #         et = sw.et
    #         bus_b = sw.element   # si et='l' es line_id, si et='b' es otro bus

    #         x = (pos_km[bus_a][0] + pos_km[bus_b][0]) / 2
    #         y = (pos_km[bus_a][1] + pos_km[bus_b][1]) / 2

    #         # dibujar línea entre buses
    #         plt.plot(
    #             [pos_km[bus_a][0], pos_km[bus_b][0]],
    #             [pos_km[bus_a][1], pos_km[bus_b][1]],
    #             color="green" if closed==True else "red", linewidth=0.3, zorder=4, linestyle="--"
    #         )
            
    #         x = (pos_km[bus_a][0] + pos_km[bus_b][0]) / 2
    #         y = (pos_km[bus_a][1] + pos_km[bus_b][1]) / 2
        
    #     # ==========================
    #     # DIBUJAR RIO
    #     # ==========================
    #     plt.plot(
    #         [1.5,-0.3],
    #         [-0.3,1.5],
    #         alpha=0.2,
    #         color="blue", linewidth=20, linestyle="-"
    #     )
        
    #     # ==========================
    #     # FORMATO FINAL
    #     # ==========================
    #     plt.title(f"IEEE123 - Estado de la red + viento (t = {hour} h)")
    #     plt.xlabel("Distancia Este–Oeste [km]")
    #     plt.ylabel("Distancia Norte–Sur [km]")
    #     plt.gca().set_aspect("equal", adjustable="box")
    #     plt.grid(alpha=0.3)
    #     plt.tight_layout()

    #     os.makedirs("figures", exist_ok=True)
    #     plt.savefig(f"figures/hour_{hour:02d}.png", dpi=200, bbox_inches="tight")
    #     plt.close()

    def plot_network(self, hour):
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
        import pandapower.topology as top
        import os

        plt.figure(figsize=(8, 6))
        net = self.net

        # =======================================
        #  GRAFO REAL DE LA RED (incluye switches)
        # =======================================
        G = top.create_nxgraph(net, respect_switches=True, include_out_of_service=False)

        # =======================================
        #  POSICIONES (convertir a km y centrar)
        # =======================================
        FT_TO_KM = 0.0003048
        pos_km = {
            bus: (float(net.bus.at[bus, "x"]) * FT_TO_KM,
                float(net.bus.at[bus, "y"]) * FT_TO_KM)
            for bus in net.bus.index
        }

        # Centrar posiciones
        x_vals = [p[0] for p in pos_km.values()]
        y_vals = [p[1] for p in pos_km.values()]
        x_mean, y_mean = np.mean(x_vals), np.mean(y_vals)
        pos_km = {b: (pos_km[b][0]-x_mean, pos_km[b][1]-y_mean) for b in pos_km}

        # =======================================
        # 1) DIBUJAR VIENTO
        # =======================================
        if hasattr(self, "last_gust_speed") and isinstance(self.last_gust_speed, np.ndarray):
            if hasattr(self, "grid_x") and hasattr(self, "grid_y"):
                X, Y = np.meshgrid(self.grid_x, self.grid_y)
            else:
                X, Y = None, None

            if X is not None:
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
        # 2) CÁLCULO CORRECTO DE BUSES ENERGIZADOS
        # =======================================
        # Slack real
        slack_bus = int(net.ext_grid.bus.values[0])

        # Buses energizados = componente conexa del slack
        if slack_bus in G:
            energized = set(nx.node_connected_component(G, slack_bus))
        else:
            energized = set()

        # =======================================
        # 3) PREPARAR ORDEN DE NODOS CORRECTO
        # =======================================
        nodes = list(G.nodes())
        pos_filtered = {n: pos_km[n] for n in nodes}

        node_colors = []
        node_sizes  = []

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

        # =======================================
        # 4) DIBUJAR NODOS
        # =======================================
        nx.draw_networkx_nodes(
            G,
            pos_filtered,
            nodelist=nodes,   # 🔥 Mantiene correspondencia 1 a 1
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="black"
        )

        # =======================================
        # 5) DIBUJAR LÍNEAS
        # =======================================
        for lid in net.line.index:
            fb = net.line.at[lid, "from_bus"]
            tb = net.line.at[lid, "to_bus"]

            if fb in pos_km and tb in pos_km:
                x0, y0 = pos_km[fb]
                x1, y1 = pos_km[tb]

                plt.plot(
                    [x0, x1], [y0, y1],
                    color="green" if self.line_status.get(lid, 1)==1 else "red",
                    linestyle="-",
                    linewidth=0.8,
                    alpha=0.8,
                )

        # =======================================
        # 6) DIBUJAR SWITCHES B-B Y B-L
        # =======================================
        for _, sw in net.switch.iterrows():
            bus_a = sw.bus
            if sw.et == "b":  # bus-bus switch
                bus_b = sw.element
            else:             # bus-line switch
                continue

            x0, y0 = pos_km[bus_a]
            x1, y1 = pos_km[bus_b]

            plt.plot(
                [x0, x1], [y0, y1],
                color="green" if sw.closed else "red",
                linewidth=0.5,
                linestyle="--",
                zorder=4
            )

        # =======================================
        # 7) RÍO (igual que tu versión)
        # =======================================
        plt.plot(
            [1.5, -0.3], [-0.3, 1.5],
            alpha=0.2, color="blue", linewidth=20
        )

        # =======================================
        # 8) FORMATOS
        # =======================================
        plt.title(f"IEEE123 - Estado de la red + viento (t = {hour} h)")
        plt.xlabel("Distancia Este–Oeste [km]")
        plt.ylabel("Distancia Norte–Sur [km]")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/hour_{hour:02d}.png", dpi=200, bbox_inches="tight")
        plt.close()
