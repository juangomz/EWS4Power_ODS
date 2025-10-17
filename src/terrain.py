import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge
from scipy.ndimage import zoom
from pyproj import CRS, Transformer # Importar pyproj para la conversión de coordenadas
import matplotlib.pyplot as plt

class MDTProcessor:
    """
    Procesa archivos MDT GeoTIFF (típicamente en UTM) para obtener una matriz
    de elevación normalizada (0 a 1) adaptada a la malla de la simulación DANA.
    """
    
    # CRS estándar para los datos de MDT del IGN
    CRS_MDT = CRS.from_epsg(25830) # ETRS89 / UTM zone 30N (típico de Alicante)

    def __init__(self, tiff_paths, target_shape=(40, 40), bbox_latlon=None):
        """
        :param tiff_paths: Lista de rutas a los archivos GeoTIFF (e.g., [913.TIF, 914.TIF])
        :param target_shape: (ny, nx) - La dimensión de la malla de la simulación (e.g., 40, 40)
        :param bbox_latlon: (lon_min, lat_min, lon_max, lat_max) en grados decimales (WGS84)
        """
        self.tiff_paths = tiff_paths
        self.target_shape = target_shape
        self.bbox_latlon = bbox_latlon
        self.dem_data = None


    def _convert_bbox_to_utm(self):
        """Convierte las coordenadas Lat/Lon del simulador a UTM (Huso 30N)."""
        if self.bbox_latlon is None:
            raise ValueError("Bounding box (lon_min, lat_min, lon_max, lat_max) es necesario.")
            
        lon_min, lat_min, lon_max, lat_max = self.bbox_latlon
        
        # Define el transformador: de WGS84 (Lat/Lon) a ETRS89/UTM 30N (IGN MDT)
        transformer = Transformer.from_crs("EPSG:4326", self.CRS_MDT, always_xy=True)
        
        # Conversión de las esquinas
        utm_x_min, utm_y_min = transformer.transform(lon_min, lat_min)
        utm_x_max, utm_y_max = transformer.transform(lon_max, lat_max)
        
        # El bbox para rasterio.merge/from_bounds debe ser (left, bottom, right, top)
        return (utm_x_min, utm_y_min, utm_x_max, utm_y_max)


    def load_and_process(self):
        """
        Carga múltiples TIFFs, los une, recorta al BBox de la simulación, remuestrea
        y normaliza la elevación.
        """
        
        # 1. Convertir BBox Lat/Lon a UTM
        bbox_utm = self._convert_bbox_to_utm()
        
        src_files_to_mosaic = []
        for path in self.tiff_paths:
            try:
                src = rasterio.open(path)
                src_files_to_mosaic.append(src)
            except rasterio.RasterioIOError as e:
                print(f"ERROR: No se pudo abrir el archivo TIFF en {path}. {e}")
                # Cerramos archivos abiertos antes de fallar
                for s in src_files_to_mosaic: s.close()
                raise FileNotFoundError(f"Archivo TIFF no encontrado o corrupto: {path}")

        # 2. Unir los TIFFs (mosaic)
        mosaic, out_transform = merge(src_files_to_mosaic, 
                                      bounds=bbox_utm, 
                                      res=5.0) # Asegura que la resolución de salida es 5m

        # 3. Recortar (El merge ya recorta al BBox, solo necesitamos la primera banda)
        data = mosaic[0]
        
        # 4. Limpieza de datos
        # Reemplazar valores NoData por el valor medio de los datos válidos
        data[data < -9999] = np.nan
        data = np.nan_to_num(data, nan=np.nanmean(data))
        
        # 5. Re-muestreo a la malla de la simulación (target_shape)
        ny_actual, nx_actual = data.shape
        zoom_factors = [self.target_shape[0] / ny_actual, self.target_shape[1] / nx_actual]
        dem_resized = zoom(data, zoom_factors, order=1) # order=1: interpolación bilineal
        
        # 6. Normalización (0 a 1)
        min_val = dem_resized.min()
        max_val = dem_resized.max()
        
        if max_val == min_val:
            # Si el área es completamente plana, devuelve una matriz de 0.5
            return np.ones(self.target_shape) * 0.5
            
        dem_normalized = (dem_resized - min_val) / (max_val - min_val)

        # Cierra todos los datasets de origen abiertos
        for src in src_files_to_mosaic:
            src.close()
            
        # Almacenar la elevación bruta real para mapeo de CTs (opcional)
        self.elev_raw = dem_resized 

        return dem_normalized
    


    # ... (El resto de la clase MDTProcessor sigue igual) ...

    def plot_terrain(self, title="Terreno Normalizado (MDT05 Real)", show=True):
        """
        Visualiza la matriz de elevación normalizada.
        """
        if self.dem_data is None:
            print("ERROR: El terreno no ha sido cargado. Ejecute load_and_process() primero.")
            return

        plt.figure(figsize=(8, 8))
        
        # Usamos imshow para visualizar la matriz. 'cmap' define los colores (terreno)
        # 'origin'='lower' es común para mapas (el origen está abajo a la izquierda)
        im = plt.imshow(self.dem_data, cmap='terrain', origin='lower',
                        extent=[self.bbox_latlon[0], self.bbox_latlon[2], 
                                self.bbox_latlon[1], self.bbox_latlon[3]])

        plt.colorbar(im, label='Elevación Normalizada (0=Bajo, 1=Alto)')
        plt.title(title)
        plt.xlabel('Longitud (grados)')
        plt.ylabel('Latitud (grados)')
        
        # Para que las zonas bajas sean fácilmente visibles
        plt.contour(self.dem_data, levels=[0.1, 0.2], colors='blue', alpha=0.5, origin='lower',
                    extent=[self.bbox_latlon[0], self.bbox_latlon[2], 
                            self.bbox_latlon[1], self.bbox_latlon[3]])
        
        if show:
            plt.show()

        return plt.gcf() # Devuelve la figura