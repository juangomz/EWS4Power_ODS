import mosaik_api, pypsa
from simuladores.logger import Logger
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import networkx as nx

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'PyPSA_Grid': {
            'public': True,
            'params': [],
            'attrs': ['line_status','wind_speed', 'ens'],
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

        n.add("Bus", "bus0", v_nom=0.4)
        n.add("Bus", "bus1", v_nom=0.4)
        n.add("Bus", "bus2", v_nom=0.4)

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

            if 'line_status' in vals:
                line_status_inputs = vals['line_status']

        print(f"🌬️  Wind speed = {wind_speed:.2f} m/s")
        print(f"⚡ Raw line_status input = {line_status_inputs}")

        # Generar el failure_map automáticamente si aún no existe
        if not self.failure_map and self.lines:
            self.failure_map = {f"FailureModel-{i}.FailureProc": lid
                                for i, lid in enumerate(self.lines.keys())}
            print("🔗 failure_map generado automáticamente:", self.failure_map)

        # --- 2️⃣ Traducir a líneas reales ---
        line_status = {}
        for src_id, status in line_status_inputs.items():
            line_id = self.failure_map.get(src_id)
            if line_id:
                line_status[line_id] = status
            else:
                print(f"⚠️  {src_id} no tiene mapeo definido, ignorado")

        # Asegurar que todas las líneas tengan estado (default=1)
        for lid in self.lines.keys():
            line_status.setdefault(lid, 1)

        print(f"🔀 Estado interpretado de líneas = {line_status}")

        # --- 3️⃣ Actualizar red ---
        for lid, status in line_status.items():
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
        
        self.plot_network(hour, line_status)

        # --- 6️⃣ Guardar CSV por hora ---
        os.makedirs("results", exist_ok=True)
        filename = f"results/hour_{hour:02d}.csv"

        fieldnames = ["hour", "wind_speed"] + list(self.lines.keys()) + ["ens"]

        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            row = {"hour": hour, "wind_speed": wind_speed, "ens": ens}
            for lid in self.lines.keys():
                row[lid] = line_status.get(lid, 1)
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
        return {self.eid: self.current}

    def plot_network(self, hour, line_status):
        """Dibuja el estado actual de la red."""
        G = self.network.graph()
        pos = nx.spring_layout(G, seed=42)  # disposición de nodos

        plt.figure(figsize=(5, 4))
        
        # Dibujar buses
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800, edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_weight='bold')

        # Dibujar líneas activas y caídas
        active_edges = [edge for edge in G.edges if line_status.get(edge[0] + edge[1], 1) == 1]
        down_edges = [edge for edge in G.edges if line_status.get(edge[0] + edge[1], 1) == 0]

        # Si tus líneas no tienen nombres tipo 'bus0bus1', usa el ID real de PyPSA:
        active_edges = []
        down_edges = []
        for lid, vals in self.lines.items():
            edge = (vals['bus0'], vals['bus1'])
            if line_status[lid] == 1:
                active_edges.append(edge)
            else:
                down_edges.append(edge)

        nx.draw_networkx_edges(G, pos, edgelist=active_edges, edge_color='green', width=2)
        nx.draw_networkx_edges(G, pos, edgelist=down_edges, edge_color='red', width=2, style='dashed')

        plt.title(f"Network status - Hour {hour}")
        plt.axis('off')

        plt.savefig(f"figures/hour_{hour:02d}.png", dpi=150)
        plt.close()


