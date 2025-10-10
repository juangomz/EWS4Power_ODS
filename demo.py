import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString
import networkx as nx
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

from src.wind_failiure import failure_probability_from_wind

# ==============================
# 1. Crear un raster sintético de viento (10x10 píxeles)
# ==============================
width = height = 100
cell_size = 1
wind_data = np.random.uniform(5, 40, size=(height, width))  # viento 5-30 m/s

wind_smooth = gaussian_filter(wind_data, sigma=10)*10-205

transform = from_origin(0, height, cell_size, cell_size)

with rasterio.open(
    "viento.tif", "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=wind_smooth.dtype,
    crs="+proj=latlong",
    transform=transform,
) as dst:
    dst.write(wind_smooth, 1)

# ==============================
# 2. Definir red de distribución (NetworkX + coordenadas)
# ==============================
G = nx.Graph()
G.add_node("Subestacion", demanda=0, pos=(5, 50))
G.add_node("A", demanda=500, pos=(20, 20))
G.add_node("B", demanda=300, pos=(30, 70))
G.add_node("C", demanda=400, pos=(70, 50))
G.add_node("D", demanda=200, pos=(80, 20))

# Aristas con geometría
edges = [
    ("Subestacion", "A", LineString([(5,50),(20,20)])),
    ("A", "B", LineString([(20,20),(30,70)])),
    ("A", "C", LineString([(20,20),(70,50)])),
    ("C", "D", LineString([(70,50),(80,20)]))
]
for u,v,geom in edges:
    G.add_edge(u, v, geometry=geom, prob_falla=0.0)

# ==============================
# 3. Función para extraer viento medio en una línea
# ==============================
def get_raster_value(geometry, raster_path):
    with rasterio.open(raster_path) as src:
        values = []
        for x, y in geometry.coords:
            row, col = src.index(x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                values.append(src.read(1)[row, col])
        return np.mean(values) if values else 0

# Asignar viento y riesgo a cada línea
for u,v,data in G.edges(data=True):
    viento = get_raster_value(data["geometry"], "viento.tif")
    data["viento_m_s"] = viento
    # Añadimos atributos extra si no existen
    data.setdefault("length_km", 2.0)     # longitud de línea
    data.setdefault("age_score", 0.4)    # edad relativa (0 nuevo, 1 muy viejo)
    data.setdefault("veg_score", 0.3)    # exposición a vegetación
    data.setdefault("precip_mm", 10.0)   # precipitación en mm
    data.setdefault("wind_gust_ms", data.get("viento_m_s", 12.0))  # usamos ráfagas si las tienes

    # Calculamos probabilidad de falla
    data["prob_falla"] = failure_probability_from_wind(data, dt_hours=2.0)

# ==============================
# 4. Función de simulación Monte Carlo
# ==============================
def simular_fallas(G):
    Gtemp = G.copy()
    for u, v, data in list(G.edges(data=True)):
        if np.random.rand() < data["prob_falla"]:
            Gtemp.remove_edge(u, v)
    # Componentes conectados a la subestación
    if "Subestacion" in Gtemp:
        conectados = nx.node_connected_component(Gtemp, "Subestacion")
    else:
        conectados = []
    ens = sum(G.nodes[n]["demanda"] for n in G.nodes if n not in conectados)
    return ens

# ==============================
# 5. Monte Carlo
# ==============================
N = 5000
resultados = [simular_fallas(G) for _ in range(N)]
ENS_esperado = np.mean(resultados)

print("Probabilidades de falla por línea:")
for u,v,data in G.edges(data=True):
    print(f"{u}-{v}: viento={data['viento_m_s']:.2f} m/s, p_falla={data['prob_falla']:.2f}")

print(f"\nENS esperado: {ENS_esperado:.2f} kW no servidos")

# ==============================
# 6. Visualización
# ==============================
# Histograma ENS
plt.hist(resultados, bins=20, color="orange", edgecolor="black")
plt.xlabel("ENS (kW no servidos)")
plt.ylabel("Frecuencia")
plt.title("Distribución de ENS (Monte Carlo)")
plt.savefig("ens_histograma.png")

# Mapa: red sobre raster de viento
fig, ax = plt.subplots()
plt.imshow(wind_smooth, extent=(0,width,0,height), origin="lower", cmap="Reds", alpha=0.5)
pos = nx.get_node_attributes(G, "pos")
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=100, ax=ax)
plt.title("Red de distribución sobre mapa de viento")
plt.colorbar(label="Viento (m/s)")
plt.savefig("red_viento.png")