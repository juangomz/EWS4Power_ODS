import rasterio
from pyproj import CRS, Transformer

# Asume que este es el archivo principal o el primero de tu lista de MDT05
FILE_PATH = 'mdt_files/PNOA_MDT05_ETRS89_HU30_0913_LID.tif' 

# 1. Abrir el archivo TIFF y leer los metadatos
with rasterio.open(FILE_PATH) as src:
    # Coordenadas UTM del borde del archivo
    utm_bounds = src.bounds 
    
    # CRS original (e.g., ETRS89 / UTM 30N)
    source_crs = src.crs 

# 2. Definir los sistemas de coordenadas para la conversión
# Destino: WGS84 (Lat/Lon) - estándar para tu simulación DANA
target_crs = CRS.from_epsg(4326) 

# 3. Crear el transformador (UTM -> Lat/Lon)
transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

# 4. Convertir las esquinas UTM a Lat/Lon
# Bounding box UTM: (left, bottom, right, top)
# Convertir (lon_min, lat_min)
lon_min_deg, lat_min_deg = transformer.transform(utm_bounds.left, utm_bounds.bottom)
# Convertir (lon_max, lat_max)
lon_max_deg, lat_max_deg = transformer.transform(utm_bounds.right, utm_bounds.top)

# 5. Mostrar los resultados
print("--- Límites Extraídos del GeoTIFF ---")
print(f"CRS de Origen: {source_crs}")
print(f"Límites UTM (m): {utm_bounds}")
print("-" * 35)
print(f"lat_min (Bottom): {lat_min_deg:.4f}")
print(f"lat_max (Top):    {lat_max_deg:.4f}")
print(f"lon_min (Left):   {lon_min_deg:.4f}")
print(f"lon_max (Right):  {lon_max_deg:.4f}")

# Los valores resultantes son los que debes usar en dana_sim.py:
# self.lat_min = lat_min_deg
# self.lat_max = lat_max_deg
# self.lon_min = lon_min_deg
# self.lon_max = lon_max_deg