import mosaik_api, pypsa
from simuladores.logger import Logger
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import networkx
import numpy as np

import logging
# 🔇 Silenciar mensajes INFO de PyPSA
logging.getLogger("pypsa").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pypsa")

data_dir = './data/red_electrica'

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'PyPSA_Grid': {
            'public': True,
            'params': [],
            'attrs': ['line_status','wind_speed', 'ens', 'line_positions'],  # ✅ solo strings
        }
    }
}

class PyPSASim(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid = 'Grid'
        self.network = pypsa.Network()
        self.logger = Logger("results/results.csv")
        self.lines = {}
        self.current = {}


    def setup_network(self):
        n = self.network
        n.set_snapshots(["now"])
        
        # --- 1️⃣ Leer CSVs ---
        buses_path = os.path.join(data_dir, "buses.csv")
        lines_path = os.path.join(data_dir, "lines.csv")
        loads_path = os.path.join(data_dir, "loads.csv")
        gens_path = os.path.join(data_dir, "generators.csv")

        buses = pd.read_csv(buses_path)
        lines = pd.read_csv(lines_path)
        loads = pd.read_csv(loads_path)
        gens = pd.read_csv(gens_path)

        
        # --- 2️⃣ Añadir Buses ---
        for _, row in buses.iterrows():
            bus_id = str(row["bus"])
            n.add("Bus",
                bus_id,
                v_nom=float(row.get("v_nom_kv", 20)),
                carrier="AC")
            n.buses.at[bus_id, "x"] = float(row["x"])
            n.buses.at[bus_id, "y"] = float(row["y"])

        # --- 3️⃣ Añadir Líneas ---
        line_data = {}
        for _, row in lines.iterrows():
            lid = str(row["line"])
            bus0, bus1 = str(row["bus0"]), str(row["bus1"])
            r = float(row.get("r_ohm", 0.1)) * float(row.get("length_km", 1.0))
            x = float(row.get("x_ohm", 0.3)) * float(row.get("length_km", 1.0))
            s_nom = float(row.get("s_max_mva", 10.0))

            n.add("Line", lid, bus0=bus0, bus1=bus1, r=r, x=x, s_nom=s_nom)
            line_data[lid] = {"bus0": bus0, "bus1": bus1, "r": r, "x": x, "s_nom": s_nom}

        # --- 4️⃣ Añadir Cargas ---
        for _, row in loads.iterrows():
            n.add("Load", str(row["load"]),
                bus=str(row["bus"]),
                p_set=float(row["p_set_mw"]),
                q_set=float(row.get("q_set_mvar", 0.0)))

        # --- 5️⃣ Añadir Generadores ---
        for _, row in gens.iterrows():
            n.add("Generator", str(row["gen"]),
                bus=str(row["bus"]),
                p_nom=float(row["p_nom_mw"]),
                control=row.get("control", "PQ"))
            
        # --- 6️⃣ Normalizar índices en PyPSA ---
        n.buses.index = n.buses.index.astype(str)
        n.lines.index = n.lines.index.astype(str)
        n.loads.index = n.loads.index.astype(str)
        n.generators.index = n.generators.index.astype(str)
            
        # ✅ Forzar carrier AC en caso de buses sin definir
        n.buses["carrier"] = "AC"

        # Guardamos la referencia local de líneas
        self.lines = line_data

        print(f"✅ Red cargada desde {data_dir}: "
            f"{len(buses)} buses, {len(lines)} líneas, {len(loads)} cargas, {len(gens)} generadores.")

        # Inicializar estado actual
        self.current = {
            'ens': 0.0,
            'num_lines': len(n.lines),
            'currents': {lid: 0.0 for lid in n.lines.index}
        }

    def init(self, sid, **sim_params):
        return META

    def create(self, num, model):
        """Crear la entidad de red PyPSA y configurar la topología si se pasa network_data."""
        print("📡 Recibido network_data desde mosaik_config.py")
        self.setup_network()

        # Crear el mapeo ahora que sí existen las líneas
        self.failure_map = {
            f"FailureModel-0.FailureProc_{i}": lid
            for i, lid in enumerate(self.lines.keys())
        }

        # print("🔗 failure_map generado automáticamente:", self.failure_map)
    
        # 💾 Inicializar current aquí mismo
        self.current = {
            'ens': 0.0,
            'num_lines': len(self.network.lines),
            'currents': {lid: 0.0 for lid in self.network.lines.index}
        }

        return [{'eid': self.eid, 'type': model, 'rel': []}]

    def step(self, time, inputs, max_advance):
        print("\n==============================")
        hour = int(time / 3600)
        print(f"⏱️  STEP t = {hour} h")

        # --- 1️⃣ Leer entradas ---
        wind_speed = 0
        line_status_inputs = {}

        if inputs:
            src = list(inputs.keys())[0]
            vals = inputs[src]

            if 'wind_speed' in vals:
                wind_speed = list(vals['wind_speed'].values())[0]
                self.last_wind_field = np.array(wind_speed)


            if 'line_status' in vals:
                line_status_inputs = vals['line_status']

        # print(f"🌬️  Wind speed = {wind_speed:.2f} m/s")
        # print(f"⚡ Raw line_status input = {line_status_inputs}")

        # Generar el failure_map automáticamente si aún no existe
        if not self.failure_map and self.lines:
            self.failure_map = {f"FailureModel-{i}.FailureProc": lid
                                for i, lid in enumerate(self.lines.keys())}
            # print("🔗 failure_map generado automáticamente:", self.failure_map)

        # --- 2️⃣ Traducir a líneas reales ---
        if not hasattr(self, "line_status_memory"):
            self.line_status_memory = {lid: 1 for lid in self.lines.keys()}
    
        for src_id, status in line_status_inputs.items():
            line_id = self.failure_map.get(src_id)
            if line_id:
                if self.line_status_memory[line_id] == 1 and status == 0:
                    # Solo se pasa de operativa → rota
                    self.line_status_memory[line_id] = 0
            else:
                print(f"⚠️  {src_id} no tiene mapeo definido, ignorado")


        # print(f"🔀 Estado interpretado de líneas = {self.line_status_memory}")

        # --- 3️⃣ Actualizar red ---
        for lid, status in self.line_status_memory.items():
            if status == 0:
                if lid in self.network.lines.index:
                    # print(f"❌ Eliminando línea {lid}")
                    self.network.remove("Line", lid)
            else:
                if lid not in self.network.lines.index and lid in self.lines:
                    params = self.lines[lid]
                    # print(f"✅ Restaurando línea {lid}")
                    self.network.add("Line", lid,
                                    bus0=params["bus0"],
                                    bus1=params["bus1"],
                                    x=params["x"],
                                    r=params["r"],
                                    s_nom=params["s_nom"])

        # --- 4️⃣ Flujo de potencia ---
        try:
            self.network.lpf()
            print("🧮 Flujo lineal ejecutado correctamente.")
        except Exception as e:
            print(f"💥 Error en lpf(): {e}")

        # --- Calcular corriente aproximada por línea ---
        currents = {}
        for lid, line in self.network.lines.iterrows():
            try:
                # Potencia activa (MW) -> convertir a kW para 0.4 kV nominal
                p = abs(self.network.lines_t.p0[lid].iloc[0]) * 1e3
                v = self.network.buses.at[line.bus0, 'v_nom'] * 1e3  # V
                i = p / (v if v > 0 else 1)  # I ≈ P/V
                currents[lid] = round(i, 3)
            except (KeyError, IndexError):
                currents[lid] = 0.0

        import networkx as nx

        # --- 5️⃣ Calcular ENS (Energy Not Supplied) de forma general ---
        G = self.network.graph()

        # Suma total esperada de carga (MW)
        expected_load = abs(self.network.loads["p_set"]).sum()
        actual_load = 0.0

        # Buscar todos los generadores conectados y sus buses
        gen_buses = set(self.network.generators["bus"])

        # Iterar sobre cada carga
        for load_id, load_row in self.network.loads.iterrows():
            bus = load_row["bus"]

            # Si hay al menos un camino desde esta carga a algún generador → está alimentada
            connected = any(nx.has_path(G, bus, gen_bus) for gen_bus in gen_buses if bus in G and gen_bus in G)

            if connected:
                # Si la red está conectada, la potencia servida es igual a la programada
                actual_load += abs(load_row["p_set"])
            else:
                # Si está aislada, se considera no servida
                pass

        # ENS = potencia esperada - potencia realmente servida
        ens = max(0.0, expected_load - actual_load)

        print(f"📊 Expected load = {expected_load:.2f}, ENS = {ens:.2f}")
        print("guardando plots...")
        self.plot_network(hour, self.line_status_memory)

        # --- 6️⃣ Guardar CSV por hora ---
        os.makedirs("results", exist_ok=True)
        filename = f"results/hour_{hour:02d}.csv"

        fieldnames = ["hour", "wind_speed"] + list(self.lines.keys()) + ["ens"]

        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            row = {"hour": hour, "wind_speed": wind_speed, "ens": ens}
            for lid in self.lines.keys():
                row[lid] = self.line_status_memory.get(lid, 1)
            writer.writerow(row)

        print(f"📝 CSV guardado -> {filename}")

        # --- 7️⃣ Actualizar salida Mosaik ---
        self.current = {
            'ens': ens,
            'currents': currents,
            'num_lines': len(self.network.lines)
        }
        return time + 3600


    def get_data(self, outputs=None):
        bus_pos = {bus: (float(self.network.buses.at[bus, 'x']),
                        float(self.network.buses.at[bus, 'y']))
                for bus in self.network.buses.index}

        line_pos = {
            lid: {
                "bus0": self.lines[lid]["bus0"],
                "bus1": self.lines[lid]["bus1"],
                "x0": float(self.network.buses.at[self.lines[lid]["bus0"], 'x']),
                "y0": float(self.network.buses.at[self.lines[lid]["bus0"], 'y']),
                "x1": float(self.network.buses.at[self.lines[lid]["bus1"], 'x']),
                "y1": float(self.network.buses.at[self.lines[lid]["bus1"], 'y']),
            }
            for lid in self.lines.keys()
        }

        return {
            self.eid: {
                'ens': self.current.get('ens', 0.0),
                'currents': self.current.get('currents', {}),
                'num_lines': len(self.lines),
                'line_positions': line_pos,  # ✅ añadido
            }
        }


    def plot_network(self, hour, line_status):
        """Dibuja el estado actual de la red sobre el mapa de viento (en lon/lat reales)."""
        import networkx as nx
        G = self.network.graph()

        # --- 1️⃣ Posiciones de buses (lon/lat reales) ---
        pos = {
            bus: (
                float(self.network.buses.at[bus, "x"]),  # lon
                float(self.network.buses.at[bus, "y"])   # lat
            )
            for bus in self.network.buses.index
        }

        plt.figure(figsize=(8, 6))

        # --- 2️⃣ Dibujar mapa de viento en coordenadas reales ---
        if hasattr(self, "last_wind_field") and isinstance(self.last_wind_field, np.ndarray):
            ny, nx = self.last_wind_field.shape

            # Obtener límites reales desde WindSim2D (lat/lon)
            if hasattr(self, "wind_grid_lon") and hasattr(self, "wind_grid_lat"):
                lon_min, lon_max = min(self.wind_grid_lon), max(self.wind_grid_lon)
                lat_min, lat_max = min(self.wind_grid_lat), max(self.wind_grid_lat)
            else:
                # fallback si no los tienes almacenados
                lon_min, lon_max = min(v[0] for v in pos.values()), max(v[0] for v in pos.values())
                lat_min, lat_max = min(v[1] for v in pos.values()), max(v[1] for v in pos.values())

            extent = [lon_min, lon_max, lat_min, lat_max]

            plt.imshow(
                self.last_wind_field,
                origin='lower',
                cmap='coolwarm',
                alpha=0.6,
                extent=extent,
                aspect='auto'
            )
            plt.colorbar(label='Wind speed [m/s]', shrink=0.7)
        else:
            print("⚠️  No hay campo de viento disponible para graficar.")

        # --- 3️⃣ Dibujar red ---
        networkx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2, edgecolors='black')
        # networkx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

        active_edges, down_edges = [], []
        for lid, vals in self.lines.items():
            edge = (vals['bus0'], vals['bus1'])
            if line_status.get(lid, 1) == 1:
                active_edges.append(edge)
            else:
                down_edges.append(edge)

        for lid, vals in self.lines.items():
            bus0, bus1 = vals["bus0"], vals["bus1"]
            if bus0 in pos and bus1 in pos:
                x0, y0 = pos[bus0]
                x1, y1 = pos[bus1]
                plt.plot([x0, x1], [y0, y1],
                        color="green" if line_status.get(lid, 1) == 1 else "red",
                        linewidth=0.8,
                        linestyle="--" if line_status.get(lid, 1) == 0 else "-",
                        alpha=0.8)


        plt.title(f"Network status - Hour {hour}")
        plt.xlabel("Longitude (°)")
        plt.ylabel("Latitude (°)")
        plt.grid(alpha=0.3)
        plt.savefig(f"figures/hour_{hour:02d}.png", dpi=200, bbox_inches="tight")
        plt.close()





