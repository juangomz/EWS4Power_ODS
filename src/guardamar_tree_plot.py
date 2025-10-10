# =======================================================
# Script: Red radial basada en calles reales (Guardamar)
# =======================================================
# Requisitos:
# pip install osmnx networkx geopandas matplotlib

import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# -----------------------------
# 1. Descargar red de calles
# -----------------------------
place_name = "Guardamar del Segura, Alicante, Spain"
print(f"Descargando red de calles para {place_name}...")
G = ox.graph_from_place(place_name, network_type="drive")

G = G.to_undirected()

# Convertir a GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)

# Asegurar que cada arista tenga un peso de longitud (en metros)
for u, v, data in G.edges(data=True):
    if "length" not in data:
        data["length"] = ox.distance.great_circle_vec(
            G.nodes[u]["y"], G.nodes[u]["x"],
            G.nodes[v]["y"], G.nodes[v]["x"]
        )

# -----------------------------
# 2. Crear árbol de expansión mínimo (radial)
# -----------------------------
print("Calculando árbol de expansión mínimo (MST)...")
T = nx.minimum_spanning_tree(G, weight="length")

# Convertir a conjuntos simples (sin keys)
edges_all = {(u, v) if u < v else (v, u) for u, v in G.edges()}
edges_tree = {(u, v) if u < v else (v, u) for u, v in T.edges()}

# Líneas abiertas (las que se deben cortar)
open_edges = edges_all - edges_tree
print(f"Total de líneas: {len(edges_all)}")
print(f"Líneas abiertas (fuera del árbol): {len(open_edges)}")


# Calcular el grado de cada nodo
degree_dict = dict(G.degree())

# Clasificar buses según grado y tipo de vía
bus_voltage = {}
for node in G.nodes():
    # Mira el grado del nodo
    deg = degree_dict[node]
    # Mira si tiene aristas principales
    main_street = any(G.edges[node, nbr, 0].get("highway") in ["primary", "secondary", "unclassified"] 
                      for nbr in G.neighbors(node))
    
    if main_street:
        bus_voltage[node] = 20.0  # MT
    else:
        bus_voltage[node] = 0.4   # BT
        
degree_dict = dict(T.degree())  # T es tu grafo radial (MST)
bus_type = {}

for node, deg in degree_dict.items():
    if deg >= 3:
        bus_type[node] = "generator"
    elif deg == 1:
        bus_type[node] = "load"
    else:
        bus_type[node] = "transit"
        
        
lines = []
for u, v, data in T.edges(data=True):
    length_km = data.get("length", 0) / 1000.0
    v_nom = max(bus_voltage[u], bus_voltage[v])

    if v_nom == 20.0:
        r_ohm = 0.25 * length_km
        x_ohm = 0.35 * length_km
        s_max = 25
    else:  # BT típica
        r_ohm = 0.6 * length_km
        x_ohm = 0.4 * length_km
        s_max = 0.5

    lines.append({
        "line": f"L_{u}_{v}",
        "bus0": u,
        "bus1": v,
        "v_nom_kv": v_nom,
        "length_km": length_km,
        "r_ohm": r_ohm,
        "x_ohm": x_ohm,
        "s_max_mva": s_max,
        "status": "active"  # Por defecto activas
    })

for line in lines:
    line["status"] = "active" if (min(line["bus0"], line["bus1"]), max(line["bus0"], line["bus1"])) in edges_tree else "open"
    
lines_df = pd.DataFrame(lines)
lines_df.to_csv("lines.csv", index=False)

buses = []
for node, data in G.nodes(data=True):
    buses.append({
        "bus": node,
        "v_nom_kv": bus_voltage[node],
        "x": data["x"],
        "y": data["y"],
        "type": bus_type[node]
    })
    
    gens = []
loads = []

for n in T.nodes():
    if bus_type[n] == "generator":
        gens.append({
            "gen": f"G_{n}",
            "bus": n,
            "p_nom_mw": 10.0,      # potencia disponible (puedes variar)
            "type": "slack"
        })
    elif bus_type[n] == "load":
        loads.append({
            "load": f"L_{n}",
            "bus": n,
            "p_set_mw": 0.5,       # consumo típico residencial
            "q_set_mvar": 0.15
        })


buses_df = pd.DataFrame(buses)
buses_df.to_csv("buses.csv", index=False)
pd.DataFrame(gens).to_csv("generators.csv", index=False)
pd.DataFrame(loads).to_csv("loads.csv", index=False)

# -----------------------------
# 3. Guardar resultados
# -----------------------------
output_dir = Path("guardamar_radial")
output_dir.mkdir(exist_ok=True)

# Guardar nodos
nodes_out = gpd.GeoDataFrame(nodes[["x", "y", "geometry"]])
nodes_out.to_file(output_dir / "guardamar_nodes.geojson", driver="GeoJSON")

# Asegurar que 'u' y 'v' son columnas (no solo índice)
edges_copy = edges.reset_index().copy()

# Añadir columna 'status' (líneas activas o abiertas)
edges_copy["status"] = edges_copy.apply(
    lambda r: "active"
    if ((r["u"], r["v"]) in edges_tree or (r["v"], r["u"]) in edges_tree)
    else "open",
    axis=1,
)

# Exportar a GeoJSON
edges_copy.to_file(output_dir / "guardamar_edges_status.geojson", driver="GeoJSON")

print("Datos guardados en carpeta 'guardamar_radial/'")

# -----------------------------
# 4. Visualización
# -----------------------------
pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}

plt.figure(figsize=(8, 8))
# Calles base
nx.draw(G, pos, node_size=5, edge_color="lightgray", alpha=0.4)

pos = {n:(G.nodes[n]['x'], G.nodes[n]['y']) for n in G.nodes()}
# nx.draw(T, pos, node_color=["red" if bus_voltage[n]==20 else "green" for n in G.nodes()],
#         node_size=10, edge_color="gray")
colors = {"generator": "red", "load": "green", "transit": "gray"}
nx.draw(
    T, pos,
    node_color=[colors[bus_type[n]] for n in T.nodes()],
    node_size=20,
    edge_color="black", width=0.8
)
# --- Título y ejes ---
plt.title("Clasificación MT (rojo) y BT (verde)")
plt.xlabel("Longitud (°)")
plt.ylabel("Latitud (°)")

# 🔹 Mantener escala igual en ambos ejes
plt.gca().set_aspect("equal", adjustable="box")

# 🔹 Mostrar cuadrícula y ejes
plt.grid(True, linestyle="--", alpha=0.5)

# Guardar mapa
map_path = output_dir / "guardamar_radial_map.png"
plt.savefig(map_path, dpi=150)

print(f"Mapa guardado en {map_path}")
