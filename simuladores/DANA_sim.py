# simuladores/dana_sim.py
import mosaik_api
import numpy as np
import os
# La clase MDTProcessor debe estar definida en el archivo src/terrain.py (o similar)
from src.terrain import MDTProcessor 

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'DANASim2D': {
            'public': True,
            'params': [
                'lat_min', 'lat_max', 'lon_min', 'lon_max', 'nx', 'ny',
                'seed', 'cells', 'bg_wind_ms', 
                'drain_rate', 'runoff_max'],
            'attrs': [
                'wind_speed', 'rain_intensity', 'flood_depth',
                'grid_lat', 'grid_lon', 'wind_shape', 'terrain'],
        }
    }
}


class DANASim2D(mosaik_api.Simulator):
    """
    Simulador DANA con terreno real.
    """
    
    # CONSTANTE: (Mantenida como recordatorio, pero ya no se accede aquí)
    # MDT_FILENAMES = [...] 

    def __init__(self):
        super().__init__(META)
        
        # FIX: Define la constante como atributo de instancia para evitar AttributeError
        self.MDT_FILENAMES = [ 
            'PNOA_MDT05_ETRS89_HU30_0913_LID.tif', # <-- CORREGIDO
            'PNOA_MDT05_ETRS89_HU30_0914_LID.tif', # <-- Asumiendo el mismo patrón
            'PNOA_MDT05_ETRS89_HU30_0937_LID.tif', # <-- Asumiendo el mismo patrón
        ]
        
        self.sid = None
        self.eid = None
        self.time = 0
        self.nx = 40
        self.ny = 40
        # self.lat_min = 38.0
        # self.lat_max = 38.12
        # self.lon_min = -0.70
        # self.lon_max = -0.60
        self.lat_min = 37.9996
        self.lat_max = 38.1671
        self.lon_min = -1.1923
        self.lon_max = -0.8487
        self.deg2km = 111.0
        self.current = None

        # parámetros (Valores ajustados para inundación visible)
        self.seed = None
        self.rng = np.random.default_rng()
        self.cells = 4
        self.bg_wind_ms = 6.0
        self.drain_rate = 0.0015 # Ajustado para inundación: 1.5 mm/h
        self.runoff_max = 0.6    # Ajustado: 60% escorrentía máxima

        # campos
        self.grid_lat = None
        self.grid_lon = None
        self.LON = None
        self.LAT = None
        self.flood = None
        self.terrain = None # Será el MDT real
        self.cells_state = []

    # --- Mosaik ---
    def init(self, sid, time_resolution=3600, **sim_params):
        self.sid = sid
        self.time_resolution = time_resolution
        return META

    def create(self, num, model,
               nx=40, ny=40, seed=None, cells=4,
               bg_wind_ms=6.0, drain_rate=0.015, runoff_max=0.2):

        self.nx, self.ny = int(nx), int(ny)
        self.grid_lat = np.linspace(self.lat_min, self.lat_max, self.ny)
        self.grid_lon = np.linspace(self.lon_min, self.lon_max, self.nx)
        self.LON, self.LAT = np.meshgrid(self.grid_lon, self.grid_lat)

        self.seed = int(seed) if seed is not None else None
        self.rng = np.random.default_rng(self.seed)
        self.cells = int(cells)
        self.bg_wind_ms = float(bg_wind_ms)
        self.drain_rate = float(drain_rate)
        self.runoff_max = float(runoff_max)

        self.eid = 'DANAField'
        self.flood = np.zeros((self.ny, self.nx))
        
        # --- CARGA DEL TERRENO REAL O USO DE SINTÉTICO ---
        MDT_FOLDER = 'mdt_files'
        
        # ESTA LÍNEA AHORA FUNCIONARÁ sin error:
        mdt_filenames = self.MDT_FILENAMES
        
        if mdt_filenames and isinstance(mdt_filenames, list) and mdt_filenames[0]:
            mdt_paths = [os.path.join(MDT_FOLDER, fn) for fn in mdt_filenames]
            
            try:
                # 1. Intentar cargar el MDT real
                bbox_latlon = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
                processor = MDTProcessor(mdt_paths, (self.ny, self.nx), bbox_latlon)
                self.terrain = processor.load_and_process()
                print(f"INFO: Terreno REAL cargado desde /{MDT_FOLDER}/. Tamaño: {self.terrain.shape}")
                
            except Exception as e:
                # 2. Si hay un fallo de I/O o procesamiento, usar el sintético
                print(f"ADVERTENCIA: Falló la carga del MDT desde /{MDT_FOLDER}/. Usando terreno sintético.")
                print(f"Error específico: {e}")
                self.terrain = self._make_synthetic_terrain()
        
        else:
            # 3. Si la lista constante está vacía, usar el sintético
            self.terrain = self._make_synthetic_terrain()
            print("INFO: Usando terreno sintético (MDT_FILENAMES está vacío).")


        # inicializa células
        self.cells_state = self._spawn_cells(t0=0.0)
        
        self.prev_wind = np.ones((self.ny, self.nx)) * self.bg_wind_ms
        self.prev_rain = np.zeros((self.ny, self.nx))

        return [{'eid': self.eid, 'type': model, 'rel': []}]

    # ... (El resto del código del simulador permanece igual) ...
    def step(self, time, inputs, max_advance):
        t_h = time / 3600.0

        # actualiza movimiento y vida de las células
        self._update_cells(t_h)

        # lluvia
        rain = self._compose_rain(t_h)

        # viento de fondo + ráfagas por precipitación
        wind = self._compose_wind(rain)

        # inundación
        self._update_flood(rain)

        self.current = {
            'wind_speed': wind,
            'rain_intensity': rain,
            'flood_depth': self.flood,
        }

        return time + self.time_resolution

    def get_data(self, outputs):
        if self.current is None:
            empty = np.zeros((self.ny, self.nx))
            return {self.eid: {
                'wind_speed': empty.tolist(),
                'rain_intensity': empty.tolist(),
                'flood_depth': empty.tolist(),
                'grid_lat': self.grid_lat.tolist() if self.grid_lat is not None else [],
                'grid_lon': self.grid_lon.tolist() if self.grid_lon is not None else [],
                'wind_shape': [0, 0],
                'terrain': empty.tolist(),
            }}

        data = self.current
        return {self.eid: {
            'wind_speed': data['wind_speed'].tolist(),
            'rain_intensity': data['rain_intensity'].tolist(),
            'flood_depth': data['flood_depth'].tolist(),
            'grid_lat': self.grid_lat.tolist(),
            'grid_lon': self.grid_lon.tolist(),
            'wind_shape': list(data['wind_speed'].shape),
            'terrain': self.terrain.tolist(),
        }}

    def _make_terrain(self):
        return self.terrain

    def _make_synthetic_terrain(self):
        # Código sintético de respaldo
        x = (self.LON - self.lon_min) / (self.lon_max - self.lon_min)
        y = (self.LAT - self.lat_min) / (self.lat_max - self.lat_min)
        hills = 0.4 + 0.2*np.sin(2*np.pi*(0.6*x + 0.4*y)) * np.cos(2*np.pi*(0.2*x - 0.6*y))
        basin = 0.2 * np.exp(-(((x-0.85)/0.15)**2 + ((y-0.25)/0.12)**2))
        return np.clip(hills - basin, 0.0, 1.0)

    def _spawn_cells(self, t0):
        cells = []
        for _ in range(self.cells):
            lon_c = self.rng.uniform(self.lon_min, self.lon_max)
            lat_c = self.rng.uniform(self.lat_min, self.lat_max)
            a_km = self.rng.uniform(2.0, 6.0)
            b_km = self.rng.uniform(1.5, 4.0)
            peak = self.rng.uniform(30, 100.0)
            life = self.rng.uniform(1.0, 3.0)  # horas
            dir_ = self.rng.uniform(0, 2*np.pi)
            cells.append({
                'lon': lon_c, 'lat': lat_c, 'a': a_km, 'b': b_km,
                'peak': peak, 't_birth': t0, 't_death': t0 + life, 'dir': dir_
            })
        return cells

    def _update_cells(self, t_h):
        alive = []
        for c in self.cells_state:
            if t_h <= c['t_death']:
                dx_km = 8.0 * np.cos(c['dir'])  # velocidad de ~8 km/h
                dy_km = 8.0 * np.sin(c['dir'])
                c['lon'] += dx_km / (self.deg2km * np.cos(np.radians(c['lat'])))
                c['lat'] += dy_km / self.deg2km
                alive.append(c)
        self.cells_state = alive
        # reemplaza células muertas
        while len(self.cells_state) < self.cells:
            self.cells_state.extend(self._spawn_cells(t0=t_h))

    def _elliptical_r(self, lon_c, lat_c, a_km, b_km):
        dx = (self.LON - lon_c) * self.deg2km * np.cos(np.radians(lat_c))
        dy = (self.LAT - lat_c) * self.deg2km
        return np.sqrt((dx/a_km)**2 + (dy/b_km)**2)

    def _compose_rain(self, t_h):
        rain = np.zeros((self.ny, self.nx))
        for c in self.cells_state:
            life = c['t_death'] - c['t_birth']
            tau = np.clip((t_h - c['t_birth']) / max(life, 1e-6), 0, 1)
            temporal = np.sin(np.pi * tau)
            r = self._elliptical_r(c['lon'], c['lat'], c['a'], c['b'])
            spatial = np.exp(-0.5 * r**2)
            rain += c['peak'] * temporal * spatial

        # ⚙️ Ruido progresivo: crece hasta 20% con el tiempo
        noise_std = min(1.0, 0.2 + 0.05 * t_h)
        rain += self.rng.normal(0, noise_std, rain.shape)

        # ✅ Suavizado temporal (memoria)
        alpha = 0.6
        if hasattr(self, "prev_rain"):
            rain = alpha * self.prev_rain + (1 - alpha) * rain
        self.prev_rain = rain.copy()

        return np.clip(rain, 0, None)

    def _compose_wind(self, rain):
        wind = self.bg_wind_ms + self.rng.normal(0, 0.3, rain.shape)

        # ⚙️ Ráfagas dependientes de lluvia
        gust_factor = np.clip(0.6 * (rain / (np.percentile(rain, 95) + 1e-6)), 0, 0.8)
        wind *= (1.0 + gust_factor)

        # ⚙️ Ruido progresivo: crece las primeras horas
        if not hasattr(self, "step_count"):
            self.step_count = 0
        self.step_count += 1
        noise_scale = min(1.0, 0.1 + 0.05 * self.step_count)
        wind += self.rng.normal(0, noise_scale, wind.shape)

        # ✅ Suavizado temporal (memoria)
        alpha = 0.7
        if hasattr(self, "prev_wind"):
            wind = alpha * self.prev_wind + (1 - alpha) * wind
        self.prev_wind = wind.copy()

        return np.clip(wind, 0, None)

    def _update_flood(self, rain):
        # Lluvia en m/h.
        rain_m_per_h = rain / 1000.0
        
        # ENTRADA: escorrentía (runoff). Más escorrentía con más lluvia.
        runoff_factor = np.clip(0.3 + 0.3 * (rain / 80.0), 0.02, self.runoff_max)
        runoff = runoff_factor * rain_m_per_h
        
        # SALIDA: drenaje (drain). Drena menos en zonas bajas (donde self.terrain es bajo, e.g., ~0).
        drain = self.drain_rate * (0.6 + 0.6 * self.terrain)
        
        # Acumulación neta (profundidad en metros) por el intervalo de tiempo (1 hora por defecto)
        dt_h = self.time_resolution / 3600.0
        self.flood += (runoff - drain) * dt_h
        self.flood = np.clip(self.flood, 0.0, 2.0)