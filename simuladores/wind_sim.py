import mosaik_api
import numpy as np

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'ClimateSim2D': {
            'public': True,
            'params': ['lat_min', 'lat_max', 'lon_min', 'lon_max', 'nx', 'ny', 'seed', 'bg_wind_ms', 'drain_rate', 'runoff_max'],
            'attrs': ['wind_speed', 'wind_shape' 'rain_intensity', 'flood_depth', 'grid_lat', 'grid_lon', 'field_shape'],
        }
    }
}


class ClimateSim2D(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.sid = None
        self.eid = None
        self.time = 0
        self.nx = 20
        self.ny = 20
        self.lat_min = 38   # Guardamar del Segura aprox
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
        self.eid = 'ClimateField'

        # Permitir override de los límites geográficos
        self.lat_min = lat_min or self.lat_min
        self.lat_max = lat_max or self.lat_max
        self.lon_min = lon_min or self.lon_min
        self.lon_max = lon_max or self.lon_max

        # Crear malla geográfica en grados
        self.grid_lat = np.linspace(self.lat_min, self.lat_max, self.ny)
        self.grid_lon = np.linspace(self.lon_min, self.lon_max, self.nx)

        return [{'eid': self.eid, 'type': model, 'rel': []}]

    # --- Simulación del viento ---
    def step(self, time, inputs, max_advance):
        self.time = time
        t_h = time / 3600.0

        # === Definir dominio ===
        LON, LAT = np.meshgrid(self.grid_lon, self.grid_lat)
        deg2km = 111.0

        # Centro del tornado (moviéndose lentamente)
        lon_c = np.mean(self.grid_lon) + 0.05 * np.sin(t_h / 2)
        lat_c = np.mean(self.grid_lat) + 0.05 * np.sin(t_h / 3)

        # === Convertir a coordenadas (km) relativas al centro ===
        dx = (LON - lon_c) * deg2km * np.cos(np.radians(lat_c))
        dy = (LAT - lat_c) * deg2km
        r = np.sqrt(dx**2 + dy**2) + 1e-6  # km

        # === 3️⃣ Perfil Rankine ===
        Rc = 1  # radio del núcleo (km)
        Vmax_peak = 35.0  # m/s (tornado severo)
        growth = np.clip(t_h / 5.0, 0, 1)  # crecimiento en las primeras 5 horas
        Vmax = Vmax_peak * growth
        V = np.where(r < Rc, Vmax * (r / Rc), Vmax * (Rc / r))

        # === 4️⃣ Dirección de rotación (ciclónica)
        u = -dy / r * V
        v =  dx / r * V

        # === 5️⃣ Magnitud y ruido
        wind_field = np.sqrt(u**2 + v**2)
        wind_field += np.random.normal(0, 1.0, wind_field.shape)
        wind_field = np.clip(wind_field, 0, None)

        # === 6️⃣ Guardar ===
        self.current = {'wind_speed': wind_field}
        self.wind_field = wind_field

        return time + self.time_resolution

    # Entrega de datos a Mosaik
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
