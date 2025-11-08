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

data_dir = './data/red_electrica'

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'PPModel': {
            'public': True,
            'params': [],
            'attrs': [
                'line_status',
                'wind_speed',
                'ens',
                'line_positions',
                'grid_x',
                'grid_y',
                'wind_shape',
            ],
        }
    },
}


class PPModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid = 'Grid'
        self.net = pp.create_empty_network()
        self.logger = Logger("results/results.csv")
        self.lines = {}
        self.line_status_memory = {}
        self.current = {}

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

        coords = pd.read_csv("feeder123/BusCoords.dat", header=None, sep=r"[,\s]+", engine="python")
        coords.columns = ["bus", "x", "y"]
        coords["bus"] = coords["bus"].astype(str).str.strip()

        bus_names = pd.unique(lines_xls[['Node A', 'Node B']].values.ravel('K'))
        for name in bus_names:
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
                bus0 = self.net.bus[self.net.bus.name == str(row['Node A']).strip()].index[0]
                bus1 = self.net.bus[self.net.bus.name == str(row['Node B']).strip()].index[0]
                length_km = row['Length (ft.)'] * 0.0003048 # pasar de pies a km

                cfg = str(row['Config.']).strip()
                # valores aproximados si no hay impedancias detalladas
                if 'ug' in cfg.lower():
                    r_ohm_per_km, x_ohm_per_km = 0.6, 0.08
                elif 'oh' in cfg.lower() or 'a' in cfg.lower():
                    r_ohm_per_km, x_ohm_per_km = 0.4, 0.05
                else:
                    r_ohm_per_km, x_ohm_per_km = 0.5, 0.06

                pp.create_line_from_parameters(
                    self.net, from_bus=bus0, to_bus=bus1,
                    length_km=length_km,
                    r_ohm_per_km=r_ohm_per_km, x_ohm_per_km=x_ohm_per_km,
                    c_nf_per_km=0.0, max_i_ka=0.2,
                    name=f"Line_{row['Node A']}_{row['Node B']}"
                )
            except Exception as e:
                print(f"⚠️  Error creando línea {row.get('Node A', '?')}-{row.get('Node B', '?')}: {e}")
                
        # Punetes entre ramas desconectadas
        bridge_pairs = [
            ("18", "35"),    # 18–35r–35  → 18–35
            ("45", "52"),    # 45–52r–52  → 45–52
            ("61", "67"),    # 61s/61r–67r–67  → 61–67
            ("97", "101"),   # 97–101r–101
            ("150", "1"),    # fuente principal
            ("160", "67"),   # subrama media
        ]

        # Incluir puentes en la red
        for a, b in bridge_pairs:
            if a in self.net.bus.name.values and b in self.net.bus.name.values:
                bus_a = self.net.bus[self.net.bus.name == a].index[0]
                bus_b = self.net.bus[self.net.bus.name == b].index[0]
                pp.create_line_from_parameters(
                    self.net, from_bus=bus_a, to_bus=bus_b,
                    length_km=0.001,     # línea corta (≈ 1 m)
                    r_ohm_per_km=0.0001, # resistencia casi nula
                    x_ohm_per_km=0.0001,
                    c_nf_per_km=0.0,
                    max_i_ka=1.0,
                    name=f"Bridge_{a}_{b}"
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

        # === Crear barra slack ===
        slack_bus = self.net.bus[self.net.bus.name == '150'].index[0] if '150' in self.net.bus.name.values else self.net.bus.index[0]
        pp.create_ext_grid(self.net, bus=slack_bus, vm_pu=1.0, name="Source")
        
        self.lines = dict(zip(self.net.line.index, self.net.line.name))
        self.line_status_memory = {lid: 1 for lid in self.lines.keys()}

        print(f"✅ Red creada: {len(self.net.bus)} buses, {len(self.net.line)} líneas, {len(self.net.load)} cargas.")
        
        print(
            f"✅ Red cargada: {len(self.net.bus)} buses, "
            f"{len(self.net.line)} líneas, "
            f"{len(self.net.load)} cargas, "
            f"{len(self.net.gen)} generadores."
        )

    # ==============================================================
    # MOSAIK HOOKS
    # ==============================================================

    def init(self, sid, **sim_params):
        return META

    def create(self, num, model):
        """Crea entidad de red y configura topología."""
        print("📡 Recibido network_data desde mosaik_config.py")
        self.setup_network()

        self.current = {
            'ens': 0.0,
            'num_lines': len(self.net.line),
            'currents': {lid: 0.0 for lid in self.lines.keys()}
        }

        return [{'eid': self.eid, 'type': model, 'rel': []}]

    # ==============================================================
    # SIMULATION STEP
    # ==============================================================

    def step(self, time, inputs, max_advance):
        print("\n==============================")
        hour = int(time / 3600)
        print(f"⏱️  STEP t = {hour} h")

        # Leer entradas
        wind_speed = 0
        line_status_inputs = {}

        if inputs:
            src = list(inputs.keys())[0]
            vals = inputs[src]

            if 'wind_speed' in vals:
                wind_speed = list(vals['wind_speed'].values())[0]
                self.last_wind_field = np.array(wind_speed)

            if 'grid_x' in vals:
                self.grid_x = np.array(list(vals['grid_x'].values())[0])
            if 'grid_y' in vals:
                self.grid_y = np.array(list(vals['grid_y'].values())[0])
            if 'wind_shape' in vals:
                self.wind_shape = tuple(list(vals['wind_shape'].values())[0])

            if 'line_status' in vals:
                line_status_inputs = vals['line_status']

        # Actualizar líneas según el modelo de fallo
        for src_id, status in line_status_inputs.items():
            line_dict = line_status_inputs[src_id]
            if line_dict is not None:
                self.line_status_memory = line_dict

        # Aplicar estado de línea
        for i, (lid, status) in enumerate(self.line_status_memory.items()):
            self.net.line.at[i, 'in_service'] = bool(status)

        # Calcular flujo DC
        try:
            pp.rundcpp(self.net)
            print("✅ DC power flow calculado correctamente.")
        except Exception as e:
            print(f"⚠️ Error en rundcpp: {e}")
            return time + 3600

        # Calcular corriente aproximada por línea
        currents = {}
        for i, row in self.net.res_line.iterrows():
            currents[i] = row.loading_percent / 100.0

        # Calcular ENS topológica
        expected_load = self.net.load.p_mw.sum()
        served_load = self.net.res_load.p_mw.sum()

        ens = max(0.0, expected_load - served_load)
        print(f"Expected: {expected_load:.3f} MW, Served: {served_load:.3f} MW, ENS: {ens:.3f} MW")

        # Guardar resultados y plot
        os.makedirs("results", exist_ok=True)
        self.save_results(hour, wind_speed, ens)
        self.plot_network(hour)

        # Preparar salida
        self.current = {
            'ens': ens,
            'currents': currents,
            'num_lines': len(self.net.line)
        }

        return time + 3600

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
            row.update({str(l): self.line_status_memory.get(l, 1) for l in self.lines.keys()})
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
                'ens': self.current.get('ens', 0.0),
                'currents': self.current.get('currents', {}),
                'num_lines': len(self.net.line),
                'line_positions': line_pos,
            }
        }

    # def plot_network(self, hour):
    #     """
    #     Plots the network over the weather, showing failed components

    #     Args:
    #         hour: instant of the simulation (h)
    #     """
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
    #     if hasattr(self, "last_wind_field") and isinstance(self.last_wind_field, np.ndarray):
    #         # Recuperar grilla de viento
    #         if hasattr(self, "grid_x") and hasattr(self, "grid_y"):
    #             X, Y = np.meshgrid(self.grid_x, self.grid_y)
    #         elif hasattr(self, "wind_grid_lon") and hasattr(self, "wind_grid_lat"):
    #             # Compatibilidad con nombres antiguos
    #             X, Y = np.meshgrid(self.wind_grid_lon, self.wind_grid_lat)
    #         else:
    #             X, Y = None, None

    #         if X is not None and Y is not None:
    #             extent = [
    #                 X.min(), X.max(),
    #                 Y.min(), Y.max(),
    #             ]
    #             plt.imshow(
    #                 self.last_wind_field,
    #                 origin="lower",
    #                 cmap="coolwarm",
    #                 alpha=0.5,
    #                 extent=extent,
    #                 vmin=0,
    #                 vmax=np.max(self.last_wind_field),
    #             )
    #             plt.colorbar(label="Velocidad del viento [m/s]", shrink=0.7)
    #         else:
    #             print("⚠️ No hay malla de viento (X, Y) para graficar.")
    #     else:
    #         print("⚠️ No hay campo de viento disponible para graficar.")

    #     # ==========================
    #     # PLOT DE LA RED
    #     # ==========================
    #     nx.draw_networkx_nodes(G, pos_km, node_color="skyblue", node_size=8, edgecolors="black")

    #     for lid in net.line.index:
    #         bus0 = net.line.at[lid, "from_bus"]
    #         bus1 = net.line.at[lid, "to_bus"]
    #         if bus0 in pos_km and bus1 in pos_km:
    #             x0, y0 = pos_km[bus0]
    #             x1, y1 = pos_km[bus1]
    #             plt.plot(
    #                 [x0, x1], [y0, y1],
    #                 color="green" if self.line_status_memory.get(lid, 1) == 1 else "red",
    #                 linewidth=0.8,
    #                 linestyle="--" if self.line_status_memory.get(lid, 1) == 0 else "-",
    #                 alpha=0.8,
    #             )

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

        # Crear grafo
        G = top.create_nxgraph(net)

        # Conversión de pies → kilómetros
        FT_TO_KM = 0.0003048
        pos_km = {
            bus: (
                float(net.bus.at[bus, "x"]) * FT_TO_KM,
                float(net.bus.at[bus, "y"]) * FT_TO_KM,
            )
            for bus in net.bus.index
        }

        # Centrar figura (coincide con el dominio del viento)
        x_vals = [v[0] for v in pos_km.values()]
        y_vals = [v[1] for v in pos_km.values()]
        x_mean, y_mean = np.mean(x_vals), np.mean(y_vals)
        pos_km = {b: (x - x_mean, y - y_mean) for b, (x, y) in pos_km.items()}

        # ==========================
        # PLOT DEL VIENTO
        # ==========================
        if hasattr(self, "last_wind_field") and isinstance(self.last_wind_field, np.ndarray):
            if hasattr(self, "grid_x") and hasattr(self, "grid_y"):
                X, Y = np.meshgrid(self.grid_x, self.grid_y)
            elif hasattr(self, "wind_grid_lon") and hasattr(self, "wind_grid_lat"):
                X, Y = np.meshgrid(self.wind_grid_lon, self.wind_grid_lat)
            else:
                X, Y = None, None

            if X is not None and Y is not None:
                extent = [X.min(), X.max(), Y.min(), Y.max()]
                plt.imshow(
                    self.last_wind_field,
                    origin="lower",
                    cmap="coolwarm",
                    alpha=0.5,
                    extent=extent,
                    vmin=0,
                    vmax=np.max(self.last_wind_field),
                )
                plt.colorbar(label="Velocidad del viento [m/s]", shrink=0.7)
            else:
                print("⚠️ No hay malla de viento (X, Y) para graficar.")
        else:
            print("⚠️ No hay campo de viento disponible para graficar.")

        # ==========================
        # 💡 COLOREAR LOS BUSES SEGÚN SI ESTÁN ENERGIZADOS
        # ==========================
        slack_bus_idx = int(net.ext_grid.bus.values[0]) if len(net.ext_grid.bus) > 0 else None

        node_colors = []
        node_sizes = []
        for bus in net.bus.index:
            if bus in net.res_bus.index:
                vm = net.res_bus.at[bus, "vm_pu"]
                if np.isnan(vm) or vm < 0.1:
                    color = "red"
                else:
                    color = "skyblue"
            else:
                color = "red"

            # 💡 Marcar el bus generador (slack)
            if bus == slack_bus_idx:
                color = "limegreen"  # verde brillante
                size = 80            # más grande
            else:
                size = 20

            node_colors.append(color)
            node_sizes.append(size)
            
        # === Etiquetas de buses (número o nombre) ===
        labels = {bus: str(bus) for bus in net.bus.index}

        # Si usas nombres en lugar de índices:
        # labels = {bus: str(net.bus.at[bus, "name"]) for bus in net.bus.index}

        nx.draw_networkx_labels(
            G,
            pos_km,
            labels=labels,
            font_size=6,         # tamaño pequeño para no saturar
            font_color="black",  # o "darkgreen" si prefieres sobre el fondo
            verticalalignment="center",
            horizontalalignment="center"
)
        # ==========================
        # PLOT DE LA RED
        # ==========================
        nx.draw_networkx_nodes(G, pos_km, node_color=node_colors, node_size=8, edgecolors="black")

        for lid in net.line.index:
            bus0 = net.line.at[lid, "from_bus"]
            bus1 = net.line.at[lid, "to_bus"]
            if bus0 in pos_km and bus1 in pos_km:
                x0, y0 = pos_km[bus0]
                x1, y1 = pos_km[bus1]
                plt.plot(
                    [x0, x1], [y0, y1],
                    color="green" if self.line_status_memory.get(lid, 1) == 1 else "red",
                    linewidth=0.8,
                    linestyle="--" if self.line_status_memory.get(lid, 1) == 0 else "-",
                    alpha=0.8,
                )
                
        # Opcional: etiqueta el generador
        if slack_bus_idx is not None:
            x_slack, y_slack = pos_km[slack_bus_idx]
            plt.text(
                x_slack, y_slack + 0.02,
                "GEN",
                color="green",
                fontsize=9,
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2")
            )

        # ==========================
        # FORMATO FINAL
        # ==========================
        plt.title(f"IEEE123 - Estado de la red + viento (t = {hour} h)")
        plt.xlabel("Distancia Este–Oeste [km]")
        plt.ylabel("Distancia Norte–Sur [km]")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/hour_{hour:02d}.png", dpi=200, bbox_inches="tight")
        plt.close()

