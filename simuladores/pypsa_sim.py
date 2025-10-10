import mosaik_api, pypsa
from simuladores.logger import Logger
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import networkx
import numpy as np

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
        
         # Coordenadas (x,y) en km o grados
        bus_positions = {
            "bus0": (2.0, 2.0),
            "bus1": (4.0, 6.0),
            "bus2": (6.0, 1.0),
        }

        for bus, pos in bus_positions.items():
            n.add("Bus", bus, v_nom=0.4)
            n.buses.at[bus, "x"] = pos[0]
            n.buses.at[bus, "y"] = pos[1]

        line_data = {
            "L1": {"bus0": "bus0", "bus1": "bus1", "x": 0.01, "r": 0.02, "s_nom": 5},
            "L2": {"bus0": "bus1", "bus1": "bus2", "x": 0.015, "r": 0.025, "s_nom": 5},
            "L3": {"bus0": "bus0", "bus1": "bus2", "x": 0.02, "r": 0.03, "s_nom": 5},
        }

        for lid, vals in line_data.items():
            n.add("Line", lid, bus0=vals["bus0"], bus1=vals["bus1"],
                x=vals["x"], r=vals["r"], s_nom=vals["s_nom"])

        self.lines = line_data

        n.add("Generator", "gen1", bus="bus0", p_nom=10.0, type="slack")
        n.add("Load", "load1", bus="bus2", p_set=0.5, q_set=0.15)

        # 💾 Inicializar estado actual (para get_data)
        self.current = {
            'ens': 0.0,
            'num_lines': len(self.network.lines),
            'currents': {lid: 0.0 for lid in self.network.lines.index}
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

        print("🔗 failure_map generado automáticamente:", self.failure_map)
    
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
        print(f"⚡ Raw line_status input = {line_status_inputs}")

        # Generar el failure_map automáticamente si aún no existe
        if not self.failure_map and self.lines:
            self.failure_map = {f"FailureModel-{i}.FailureProc": lid
                                for i, lid in enumerate(self.lines.keys())}
            print("🔗 failure_map generado automáticamente:", self.failure_map)

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


        print(f"🔀 Estado interpretado de líneas = {self.line_status_memory}")

        # --- 3️⃣ Actualizar red ---
        for lid, status in self.line_status_memory.items():
            if status == 0:
                if lid in self.network.lines.index:
                    print(f"❌ Eliminando línea {lid}")
                    self.network.remove("Line", lid)
            else:
                if lid not in self.network.lines.index and lid in self.lines:
                    params = self.lines[lid]
                    print(f"✅ Restaurando línea {lid}")
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

        # --- 5️⃣ Calcular ENS ---
        expected_load = abs(self.network.loads.at['load1', 'p_set'])
        actual_load = 0.0

        G = self.network.graph()
        connected = nx.has_path(G, "bus0", "bus2")

        if connected:
            try:
                actual_load = abs(self.network.loads_t.p['load1'].iloc[0])
            except (KeyError, IndexError):
                actual_load = 0.0
            ens = max(0.0, expected_load - actual_load)
            print(f"🔗 Buses conectados → flujo activo = {actual_load:.2f}")
        else:
            ens = expected_load
            print("🚫 Buses desconectados → toda la carga se considera no servida")

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
        """Dibuja el estado actual de la red sobre el mapa de viento."""
        G = self.network.graph()

        # ✅ Posiciones fijas (geométricas)
        pos = {
            bus: (
                float(self.network.buses.at[bus, "x"]),
                float(self.network.buses.at[bus, "y"])
            )
            for bus in self.network.buses.index
        }

        plt.figure(figsize=(6, 5))

        # === 🌀 Dibujar mapa de viento (si existe) ===
        if hasattr(self, "last_wind_field") and isinstance(self.last_wind_field, np.ndarray):
            # Crear un eje de coordenadas consistente con el grid del viento
            ny, nx = self.last_wind_field.shape
            extent = [0, nx - 1, 0, ny - 1]  # ajusta a tus coordenadas reales si las tienes
            plt.imshow(self.last_wind_field, origin='lower', cmap='coolwarm', alpha=0.6, extent=extent)
            plt.colorbar(label='Wind speed [m/s]', shrink=0.7)
        else:
            print("⚠️  No hay campo de viento disponible para graficar.")

        # === ⚡ Dibujar red ===
        networkx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800, edgecolors='black')
        networkx.draw_networkx_labels(G, pos, font_weight='bold')

        active_edges, down_edges = [], []
        for lid, vals in self.lines.items():
            edge = (vals['bus0'], vals['bus1'])
            if line_status.get(lid, 1) == 1:
                active_edges.append(edge)
            else:
                down_edges.append(edge)

        networkx.draw_networkx_edges(G, pos, edgelist=active_edges, edge_color='green', width=2)
        networkx.draw_networkx_edges(G, pos, edgelist=down_edges, edge_color='red', width=2, style='dashed')

        plt.title(f"Network status - Hour {hour}")
        plt.axis('equal')
        plt.axis('off')

        plt.savefig(f"figures/hour_{hour:02d}.png", dpi=150)
        plt.close()




