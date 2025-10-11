import mosaik_api
import numpy as np

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'WindSim2D': {
            'public': True,
            'params': ['lat_min', 'lat_max', 'lon_min', 'lon_max', 'nx', 'ny'],
            'attrs': ['wind_speed', 'grid_lat', 'grid_lon', 'wind_shape'],
        }
    }
}


class WindSim2D(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.sid = None
        self.eid = None
        self.time = 0
        self.nx = 20
        self.ny = 20
        self.lat_min = 38.05   # 🔹 Guardamar del Segura aprox
        self.lat_max = 38.12
        self.lon_min = -0.70
        self.lon_max = -0.60
        self.current = None
        self.grid_lat = None
        self.grid_lon = None

    # --- Inicialización del simulador ---
    def init(self, sid, time_resolution=3600, **sim_params):
        self.sid = sid
        self.time_resolution = time_resolution
        return META

    # --- Crear entidad WindField ---
    def create(self, num, model, lat_min=None, lat_max=None, lon_min=None, lon_max=None, nx=20, ny=20):
        self.nx = nx
        self.ny = ny
        self.eid = 'WindField'

        # ✅ Permitir override de los límites geográficos
        self.lat_min = lat_min or self.lat_min
        self.lat_max = lat_max or self.lat_max
        self.lon_min = lon_min or self.lon_min
        self.lon_max = lon_max or self.lon_max

        # ✅ Crear malla geográfica en grados
        self.grid_lat = np.linspace(self.lat_min, self.lat_max, self.ny)
        self.grid_lon = np.linspace(self.lon_min, self.lon_max, self.nx)

        return [{'eid': self.eid, 'type': model, 'rel': []}]

    # --- Simulación del viento ---
    def step(self, time, inputs, max_advance):
        self.time = time

        # 🔹 Base diaria sinusoidal
        base_wind = 8 + 4 * np.sin(2 * np.pi * time / (24 * 3600))  # m/s

        # 🔹 Variación espacial suave
        LON, LAT = np.meshgrid(self.grid_lon, self.grid_lat)
        spatial_pattern = np.sin(5 * (LON - self.lon_min)) * np.cos(5 * (LAT - self.lat_min))

        # 🔹 Ruido local (pequeña turbulencia)
        noise = np.random.normal(0, 0.4, (self.ny, self.nx))

        # Campo de viento combinado
        wind_field = base_wind + 2 * spatial_pattern + noise

        # Guardar
        self.current = {'wind_speed': wind_field}

        return time + self.time_resolution

    # --- Entrega de datos a Mosaik ---
    def get_data(self, outputs):
        if self.current is None:
            return {self.eid: {
                'wind_speed': [],
                'grid_lat': [],
                'grid_lon': [],
                'wind_shape': [0, 0],
            }}

        return {self.eid: {
            'wind_speed': self.current['wind_speed'].tolist(),
            'grid_lat': self.grid_lat.tolist(),
            'grid_lon': self.grid_lon.tolist(),
            'wind_shape': list(self.current['wind_speed'].shape),
        }}
