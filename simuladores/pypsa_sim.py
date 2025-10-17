import mosaik_api, pypsa
from simuladores.logger import Logger
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
import networkx as nx
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
            'attrs': ['line_status','wind_speed', 'ens', 'line_positions', 'grid_lon', 'grid_lat', 'wind_shape', 'flood_depth', 'rain_intensity', 'terrain'],  # ✅ solo strings
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
        
        # --- Leer CSVs ---
        buses_path = os.path.join(data_dir, "buses.csv")
        lines_path = os.path.join(data_dir, "lines.csv")
        loads_path = os.path.join(data_dir, "loads.csv")
        gens_path = os.path.join(data_dir, "generators.csv")
        
        # --- Buses ---
        
        buses_df = pd.read_csv(buses_path)
        buses_df["bus"] = buses_df["bus"].astype(str).str.strip()
        buses_df.set_index("bus", inplace=True)
        buses_df.rename(columns={"v_nom_kv": "v_nom"}, inplace=True)

        # --- Lineas ---
        lines_df = pd.read_csv(lines_path)
        for col in ["line", "bus0", "bus1"]:
            if col in lines_df.columns:
                lines_df[col] = lines_df[col].astype(str).str.strip()
        lines_df.set_index("line", inplace=True)
        lines_df.rename(columns={"r_ohm": "r", "x_ohm": "x", "s_max_mva": "s_nom"}, inplace=True)

        # --- Cargas ---
        loads_df = pd.read_csv(loads_path)
        for col in ["load", "bus"]:
            if col in loads_df.columns:
                loads_df[col] = loads_df[col].astype(str).str.strip()
        loads_df.set_index("load", inplace=True)
        loads_df.rename(columns={"p_set_mw": "p_set", "q_set_mvar": "q_set"}, inplace=True)

        # --- Generadores ---
        gens_df = pd.read_csv(gens_path)
        for col in ["gen", "bus"]:
            if col in gens_df.columns:
                gens_df[col] = gens_df[col].astype(str).str.strip()
        gens_df.set_index("gen", inplace=True)
        gens_df.rename(columns={"p_nom_mw": "p_nom"}, inplace=True)

        # --- Importar masivamente en PyPSA ---
        n.import_components_from_dataframe(buses_df, "Bus")
        n.import_components_from_dataframe(lines_df, "Line")
        n.import_components_from_dataframe(loads_df, "Load")
        n.import_components_from_dataframe(gens_df, "Generator")

        # --- Asegurar coherencia de tipos dentro de PyPSA ---
        n.buses.index = n.buses.index.astype(str)
        n.lines.index = n.lines.index.astype(str)
        n.loads.index = n.loads.index.astype(str)
        n.generators.index = n.generators.index.astype(str)

        self.lines = lines_df.to_dict(orient="index")
        self.current = {'ens': 0.0, 'num_lines': len(n.lines), 'currents': {lid: 0.0 for lid in n.lines.index}}

        print(f"Red cargada (bulk import): {len(n.buses)} buses, {len(n.lines)} líneas, {len(n.loads)} cargas, {len(n.generators)} generadores.")

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
            
            if 'grid_lon' in vals:
                self.grid_lon = np.array(list(vals['grid_lon'].values())[0])
            if 'grid_lat' in vals:
                self.grid_lat = np.array(list(vals['grid_lat'].values())[0])
            if 'wind_shape' in vals:
                self.wind_shape = tuple(list(vals['wind_shape'].values())[0])
    
            if 'line_status' in vals:
                line_status_inputs = vals['line_status']
                
            if 'rain_intensity' in vals:
                rain_intensity = list(vals['rain_intensity'].values())[0]
                self.last_rain_field = np.array(rain_intensity)
            
            if 'flood_depth' in vals:
                flood_depth = list(vals['flood_depth'].values())[0]
                self.last_flood_field = np.array(flood_depth)
                
            if 'terrain' in vals:
                terrain_data = list(vals['terrain'].values())[0]
                # Lo almacenamos, asumiendo que tiene la misma forma que los campos climáticos
                self.terrain_normalized = np.array(terrain_data).reshape(self.wind_shape)

        # --- 2️⃣ Traducir a líneas reales ---
        if not hasattr(self, "line_status_memory"):
            self.line_status_memory = {lid: 1 for lid in self.lines.keys()}

        for src_id, status in line_status_inputs.items():
            line_id = self.failure_map.get(src_id)
            if line_id:
                if self.line_status_memory[line_id] == 1 and status == 0:
                    self.line_status_memory[line_id] = 0
            else:
                print(f"{src_id} no tiene mapeo definido, ignorado")

        # --- 3️⃣ Actualizar red ---
        for lid, status in self.line_status_memory.items():
            if status == 0:
                if lid in self.network.lines.index:
                    self.network.remove("Line", lid)
            else:
                if lid not in self.network.lines.index and lid in self.lines:
                    params = self.lines[lid]
                    self.network.add("Line", lid,
                                    bus0=params["bus0"],
                                    bus1=params["bus1"],
                                    x=params["x"],
                                    r=params["r"],
                                    s_nom=params["s_nom"])

        # Flujo de potencia y BODF
        # Inicializar BODF una sola vez (flujo base)
        if not hasattr(self, "BODF"):
            print("Calculando BODF base inicial...")
            self.network.lpf()
            sn = self.network.sub_networks.obj.iloc[0]
            sn.calculate_BODF()
            self.BODF = sn.BODF
            self.base_flows = self.network.lines_t.p0.loc["now"].copy()

        if not hasattr(self, "_last_active_lines"):
            self._last_active_lines = set(self.network.lines.index)

        current_active = set(self.network.lines.index)

        if current_active != self._last_active_lines:
            failed_lines = self._last_active_lines - current_active
            repaired_lines = current_active - self._last_active_lines

            # Recalcular si hay demasiados fallos (más del 5 %)
            if len(failed_lines) / len(self.lines) > 0.05:
                print("Muchos fallos simultáneos → recalculando LPF + nueva BODF.")
                try:
                    self.network.lpf()
                    sn = self.network.sub_networks.obj.iloc[0]
                    sn.calculate_BODF()
                    self.BODF = sn.BODF
                    self.base_flows = self.network.lines_t.p0.loc["now"].copy()
                except Exception as e:
                    print(f"Error al recalcular LPF/BODF: {e}")
                self._last_active_lines = current_active.copy()

            elif failed_lines and hasattr(self, "BODF"):
                print(f"Se detectaron fallos en {len(failed_lines)} líneas → aplicando BODF...")
                f_before = self.base_flows.copy()
                warned = 0

                for lid in failed_lines:
                    try:
                        if lid in self.lines:
                            k = list(self.lines.keys()).index(lid)
                        else:
                            continue

                        if k >= self.BODF.shape[1]:
                            continue

                        delta_f = self.BODF[:, k] * f_before.get(lid, 0.0)
                        f_before += delta_f
                    except Exception as e:
                        if warned < 3:
                            print(f"No se pudo aplicar BODF a {lid}: {e}")
                        warned += 1

                # Actualizar los flujos solo de líneas activas
                active_lines = self.network.lines.index
                valid_flows = f_before.reindex(active_lines, fill_value=0.0)
                self.network.lines_t.p0.loc["now"] = valid_flows
                self.base_flows = f_before.copy()
                print("Flujo actualizado con BODF (sin recalcular LPF completo).")

            else:
                print("Cambios mayores → recalculando flujo completo (LPF)...")
                try:
                    self.network.lpf()
                    sn = self.network.sub_networks.obj[0]
                    sn.calculate_BODF()
                    self.BODF = sn.BODF
                    self.base_flows = self.network.lines_t.p0.loc["now"].copy()
                    print("Flujo y BODF recalculados.")
                except Exception as e:
                    print(f"Error al recalcular LPF/BODF: {e}")

            self._last_active_lines = current_active.copy()
        else:
            print("⏸️ Sin cambios topológicos → se omite recalculo de flujo.")

        # Calculo de Corriente REVISAR!!!
        currents = {}
        for lid, line in self.network.lines.iterrows():
            try:
                p = abs(self.network.lines_t.p0[lid].iloc[0]) * 1e3
                v = self.network.buses.at[line.bus0, 'v_nom'] * 1e3
                i = p / (v if v > 0 else 1)
                currents[lid] = round(i, 3)
            except (KeyError, IndexError):
                currents[lid] = 0.0

        # Caluclo de ENS
        G = self.network.graph()
        expected_load = abs(self.network.loads["p_set"]).sum()
        actual_load = 0.0

        for component in nx.connected_components(G):
            sub_buses = set(component)
            loads_sub = self.network.loads[self.network.loads["bus"].isin(sub_buses)]
            gens_sub = self.network.generators[self.network.generators["bus"].isin(sub_buses)]

            load_sum = abs(loads_sub["p_set"]).sum()
            gen_sum = gens_sub["p_nom"].sum()

            if gen_sum >= load_sum:
                actual_load += load_sum
            else:
                actual_load += gen_sum

        ens = max(0.0, expected_load - actual_load)

        print(f"Expected Load = {expected_load:.2f}, ENS = {ens:.2f}")
        print("guardando plots...")
        self.plot_network(hour, self.line_status_memory)

        # Guardar CSV de resultados
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

        # Preparación salida para mosaik
        self.current = {
            'ens': ens,
            'currents': currents,
            'num_lines': len(self.network.lines)
        }

        return time + 3600

    # Entrega de datos a Mosaik
    def get_data(self, outputs=None):
        # Posiciones de líneas para FailureModel
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
                'line_positions': line_pos,
            }
        }
        
    # def plot_network(self, hour, line_status):
    #     """Dibuja el estado actual de la red sobre el mapa de viento (en coordenadas físicas km)."""

    #     G = self.network.graph()

    #     # Posiciones de buses
    #     pos = {
    #         bus: (
    #             float(self.network.buses.at[bus, "x"]),  # lon
    #             float(self.network.buses.at[bus, "y"])   # lat
    #         )
    #         for bus in self.network.buses.index
    #     }

    #     plt.figure(figsize=(8, 6))

    #     # Dibujar mapa de viento en km
    #     if hasattr(self, "last_wind_field") and isinstance(self.last_wind_field, np.ndarray):

    #         # Obtener límites del grid (lon/lat)
    #         if hasattr(self, "wind_grid_lon") and hasattr(self, "wind_grid_lat"):
    #             lon_min, lon_max = min(self.wind_grid_lon), max(self.wind_grid_lon)
    #             lat_min, lat_max = min(self.wind_grid_lat), max(self.wind_grid_lat)
    #         else:
    #             lon_min, lon_max = min(v[0] for v in pos.values()), max(v[0] for v in pos.values())
    #             lat_min, lat_max = min(v[1] for v in pos.values()), max(v[1] for v in pos.values())

    #         # Conversión a coordenadas físicas (km)
    #         deg2km = 111.0
    #         lon_ref = np.mean([lon_min, lon_max])
    #         lat_ref = np.mean([lat_min, lat_max])

    #         x_km = (np.array(self.grid_lon) - lon_ref) * deg2km * np.cos(np.radians(lat_ref))
    #         y_km = (np.array(self.grid_lat) - lat_ref) * deg2km
    #         extent_km = [x_km.min(), x_km.max(), y_km.min(), y_km.max()]

    #         # Plot del campo de viento
    #         plt.imshow(
    #             self.last_wind_field,
    #             origin='lower',
    #             cmap='plasma',
    #             alpha=0.6,
    #             extent=extent_km,
    #             vmin=0,
    #             vmax=20,
    #         )
    #         plt.colorbar(label='Wind speed [m/s]', shrink=0.7)

    #     else:
    #         print("No hay campo de viento disponible para graficar.")

    #     # Convertir posiciones de buses a km 
    #     deg2km = 111.0
    #     pos_km = {
    #         bus: (
    #             (v[0] - lon_ref) * deg2km * np.cos(np.radians(lat_ref)),
    #             (v[1] - lat_ref) * deg2km,
    #         )
    #         for bus, v in pos.items()
    #     }

    #     #  Dibujar red eléctrica
    #     nx.draw_networkx_nodes(G, pos_km, node_color='skyblue', node_size=2, edgecolors='black')

    #     for lid, vals in self.lines.items():
    #         bus0, bus1 = vals["bus0"], vals["bus1"]
    #         if bus0 in pos_km and bus1 in pos_km:
    #             x0, y0 = pos_km[bus0]
    #             x1, y1 = pos_km[bus1]
    #             plt.plot(
    #                 [x0, x1], [y0, y1],
    #                 color="green" if line_status.get(lid, 1) == 1 else "red",
    #                 linewidth=0.8,
    #                 linestyle="--" if line_status.get(lid, 1) == 0 else "-",
    #                 alpha=0.8,
    #             )

    #     # -Formatear y guardar plots
    #     plt.title(f"Network status - Hour {hour}")
    #     plt.xlabel("Distancia Este–Oeste [km]")
    #     plt.ylabel("Distancia Norte–Sur [km]")
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     plt.grid(alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig(f"figures/hour_{hour:02d}.png", dpi=200, bbox_inches="tight")
    #     plt.close()
    
    # def plot_network(self, hour, line_status):
    #     """Dibuja la red eléctrica sobre los campos climáticos (viento, lluvia, inundación)."""

    #     import matplotlib.pyplot as plt
    #     import networkx as nx
    #     import numpy as np

    #     G = self.network.graph()

    #     # Posiciones de buses
    #     pos = {
    #         bus: (
    #             float(self.network.buses.at[bus, "x"]),  # lon
    #             float(self.network.buses.at[bus, "y"])   # lat
    #         )
    #         for bus in self.network.buses.index
    #     }

    #     # Comprobar que existen los campos climáticos
    #     if not (hasattr(self, "last_wind_field") and isinstance(self.last_wind_field, np.ndarray)):
    #         print("⚠️ No hay campo de viento disponible para graficar.")
    #         return

    #     # Obtener límites del grid (lon/lat)
    #     if hasattr(self, "wind_grid_lon") and hasattr(self, "wind_grid_lat"):
    #         lon_min, lon_max = min(self.wind_grid_lon), max(self.wind_grid_lon)
    #         lat_min, lat_max = min(self.wind_grid_lat), max(self.wind_grid_lat)
    #     else:
    #         lon_min, lon_max = min(v[0] for v in pos.values()), max(v[0] for v in pos.values())
    #         lat_min, lat_max = min(v[1] for v in pos.values()), max(v[1] for v in pos.values())

    #     # Conversión a coordenadas físicas (km)
    #     deg2km = 111.0
    #     lon_ref = np.mean([lon_min, lon_max])
    #     lat_ref = np.mean([lat_min, lat_max])
    #     x_km = (np.array(self.grid_lon) - lon_ref) * deg2km * np.cos(np.radians(lat_ref))
    #     y_km = (np.array(self.grid_lat) - lat_ref) * deg2km
    #     extent_km = [x_km.min(), x_km.max(), y_km.min(), y_km.max()]

    #     # Campos climáticos
    #     fields = {
    #         "Wind speed [m/s]": getattr(self, "last_wind_field", None),
    #         "Rain intensity [mm/h]": getattr(self, "last_rain_field", None),
    #         "Flood depth [m]": getattr(self, "last_flood_field", None),
    #     }
    #     cmaps = {
    #         "Wind speed [m/s]": "plasma",
    #         "Rain intensity [mm/h]": "Blues",
    #         "Flood depth [m]": "GnBu",
    #     }
    #     vmins = {"Wind speed [m/s]": 0, "Rain intensity [mm/h]": 0, "Flood depth [m]": 0}
    #     vmaxs = {"Wind speed [m/s]": 30, "Rain intensity [mm/h]": 80, "Flood depth [m]": 1.5}

    #     # Crear figura con 3 subplots
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    #     for ax, (title, field) in zip(axes, fields.items()):
    #         if field is None:
    #             ax.text(0.5, 0.5, f"No data for {title}", ha="center", va="center", color="gray")
    #             ax.axis("off")
    #             continue

    #         im = ax.imshow(
    #             field,
    #             origin="lower",
    #             cmap=cmaps[title],
    #             alpha=0.75,
    #             extent=extent_km,
    #             vmin=vmins[title],
    #             vmax=vmaxs[title],
    #         )
    #         ax.set_title(title, fontsize=10)
    #         ax.set_xlabel("Distancia E-O [km]")
    #         ax.set_ylabel("Distancia N-S [km]")
    #         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    #         # Convertir posiciones de buses a km
    #         pos_km = {
    #             bus: (
    #                 (v[0] - lon_ref) * deg2km * np.cos(np.radians(lat_ref)),
    #                 (v[1] - lat_ref) * deg2km,
    #             )
    #             for bus, v in pos.items()
    #         }

    #         # Dibujar red
    #         nx.draw_networkx_nodes(G, pos_km, node_color='skyblue', node_size=8, ax=ax, edgecolors='black')
    #         for lid, vals in self.lines.items():
    #             bus0, bus1 = vals["bus0"], vals["bus1"]
    #             if bus0 in pos_km and bus1 in pos_km:
    #                 x0, y0 = pos_km[bus0]
    #                 x1, y1 = pos_km[bus1]
    #                 ax.plot(
    #                     [x0, x1], [y0, y1],
    #                     color="green" if line_status.get(lid, 1) == 1 else "red",
    #                     linewidth=0.8,
    #                     linestyle="--" if line_status.get(lid, 1) == 0 else "-",
    #                     alpha=0.8,
    #                 )

    #     plt.suptitle(f"Red eléctrica y campos climáticos — hora {hour}", fontsize=12)
    #     plt.savefig(f"figures/hour_{hour:02d}_climate.png")
    #     plt.close()

    def plot_network(self, hour, line_status):
        """Dibuja la red eléctrica sobre los campos climáticos (viento, lluvia, TERRENO + INUNDACIÓN)."""

        # Importaciones locales (como ya estaban)
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np

        G = self.network.graph()

        # Posiciones de buses (asumiendo que las coordenadas 'x' y 'y' en PyPSA son Lat/Lon)
        pos = {
            bus: (
                float(self.network.buses.at[bus, "x"]), # lon
                float(self.network.buses.at[bus, "y"])  # lat
            )
            for bus in self.network.buses.index
        }

        # Verificación de datos
        if not (hasattr(self, "last_wind_field") and hasattr(self, "terrain_normalized") and 
                hasattr(self, "grid_lon") and hasattr(self, "grid_lat")):
             print("⚠️ No hay campos climáticos/terreno completos (grid_lon, grid_lat) disponibles para graficar.")
             return

        # =================================================================
        # CRÍTICO: ALINEACIÓN DEL SISTEMA DE COORDENADAS (Lat/Lon -> KM)
        # =================================================================
        deg2km = 111.0
        
        # 1. Obtener límites de la MALLA DANA (fuente de verdad del mapa)
        lon_min_dana, lon_max_dana = np.min(self.grid_lon), np.max(self.grid_lon)
        lat_min_dana, lat_max_dana = np.min(self.grid_lat), np.max(self.grid_lat)
        
        # 2. Definir REFERENCIA ÚNICA: El centro de la malla DANA
        lon_ref = np.mean([lon_min_dana, lon_max_dana])
        lat_ref = np.mean([lat_min_dana, lat_max_dana])
        
        # 3. Conversión de la MALLA DANA a kilómetros (para el imshow)
        x_km = (self.grid_lon - lon_ref) * deg2km * np.cos(np.radians(lat_ref))
        y_km = (self.grid_lat - lat_ref) * deg2km
        extent_km = [x_km.min(), x_km.max(), y_km.min(), y_km.max()]

        # 4. Convertir posiciones de BUSES a kilómetros (USANDO LA MISMA REFERENCIA DANA)
        pos_km = {
            bus: (
                # X (Longitud)
                (v[0] - lon_ref) * deg2km * np.cos(np.radians(lat_ref)),
                # Y (Latitud)
                (v[1] - lat_ref) * deg2km,
            )
            for bus, v in pos.items()
        }

        
        # ⚠️ Cambiamos el nombre del tercer campo a "Terrain + Flood"
        fields = {
            "Wind speed [m/s]": {"data": getattr(self, "last_wind_field", None), "cmap": "plasma", "vmin": 0, "vmax": 30},
            "Rain intensity [mm/h]": {"data": getattr(self, "last_rain_field", None), "cmap": "Blues", "vmin": 0, "vmax": 80},
            "Terrain + Flood [m]": {"data": self.terrain_normalized, "cmap": "YlGn", "vmin": 0, "vmax": 1} # Capa base
        }
        
        # Crear figura con 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

        for i, (title, f_params) in enumerate(fields.items()):
            ax = axes[i]
            field = f_params["data"]
            
            if field is None:
                ax.text(0.5, 0.5, f"No data for {title}", ha="center", va="center", color="gray")
                ax.axis("off")
                continue

            # ----------------------------------------------------
            # LÓGICA ESPECÍFICA PARA EL TERCER PLOT (TERRENO + INUNDACIÓN)
            # ----------------------------------------------------
            if title == "Terrain + Flood [m]":
                # 1. Dibujar la capa base de Terreno (Elevación Normalizada)
                im_terrain = ax.imshow(
                    field,
                    origin="lower",
                    cmap=f_params["cmap"], # 'YlGn'
                    extent=extent_km,
                    vmin=0, vmax=1,
                    alpha=0.9
                )
                
                # Crear Colorbar para el Terreno
                plt.colorbar(im_terrain, ax=ax, label='Elevación Normalizada (0=Bajo)', shrink=0.7)
                
                # 2. Superponer la Inundación (Flood Depth)
                if hasattr(self, "last_flood_field") and self.last_flood_field is not None:
                    flood_visual = np.where(self.last_flood_field > 0.05, self.last_flood_field, np.nan)
                    im_flood = ax.imshow(
                        flood_visual,
                        origin="lower",
                        cmap="Blues", # Mapa de color azul para el agua
                        alpha=0.6,
                        extent=extent_km,
                        vmin=0.05,
                        vmax=1.5,
                    )
                    # Colorbar para la Inundación (Depth)
                    plt.colorbar(im_flood, ax=ax, label='Flood Depth [m]', shrink=0.7)

            # LÓGICA GENERAL PARA OTROS PLOTS (Viento y Lluvia)
            else:
                im = ax.imshow(
                    field,
                    origin="lower",
                    cmap=f_params["cmap"],
                    alpha=0.75,
                    extent=extent_km,
                    vmin=f_params["vmin"],
                    vmax=f_params["vmax"],
                )
                plt.colorbar(im, ax=ax, shrink=0.7)

            # Dibujar red (igual para todos los subplots)
            nx.draw_networkx_nodes(G, pos_km, node_color='skyblue', node_size=8, ax=ax, edgecolors='black')
            for lid, vals in self.lines.items():
                bus0, bus1 = vals["bus0"], vals["bus1"]
                if bus0 in pos_km and bus1 in pos_km:
                    x0, y0 = pos_km[bus0]
                    x1, y1 = pos_km[bus1]
                    ax.plot(
                        [x0, x1], [y0, y1],
                        color="green" if line_status.get(lid, 1) == 1 else "red",
                        linewidth=0.8,
                        linestyle="--" if line_status.get(lid, 1) == 0 else "-",
                        alpha=0.8,
                    )
                    
            # Formateo
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Distancia E-O [km]")
            ax.set_ylabel("Distancia N-S [km]")
            ax.set_aspect('equal', adjustable='box')


        plt.suptitle(f"Red eléctrica y campos climáticos — hora {hour}", fontsize=14)
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/hour_{hour:02d}_climate.png", dpi=200)
        plt.close()