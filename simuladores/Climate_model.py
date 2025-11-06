import mosaik_api
import numpy as np

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'ClimateModel': {
            'public': True,
            'params': ['x_min', 'x_max', 'y_min', 'y_max', 'nx', 'ny'],
            'attrs': ['wind_speed', 'grid_x', 'grid_y', 'wind_shape'],
        }
    }
}


class ClimateModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.sid = None
        self.eid = None
        self.time = 0
        self.nx = 20
        self.ny = 20
        # 🔹 Dominio centrado en la red (km)
        self.x_min, self.x_max = -1.5, 1.5
        self.y_min, self.y_max = -1.0, 1.0
        self.grid_x = None
        self.grid_y = None
        self.current = None

    def init(self, sid, time_resolution=3600, **sim_params):
        self.sid = sid
        self.time_resolution = time_resolution
        return META

    def create(self, num, model, x_min=None, x_max=None, y_min=None, y_max=None, nx=20, ny=20):
        self.nx, self.ny = nx, ny
        self.eid = 'WindField'

        # Permitir override
        self.x_min = x_min or self.x_min
        self.x_max = x_max or self.x_max
        self.y_min = y_min or self.y_min
        self.y_max = y_max or self.y_max

        # Crear malla en km (centrada en red)
        self.grid_x = np.linspace(self.x_min, self.x_max, self.nx)
        self.grid_y = np.linspace(self.y_min, self.y_max, self.ny)

        return [{'eid': self.eid, 'type': model, 'rel': []}]

    def step(self, time, inputs, max_advance):
        self.time = time
        t_h = time / 3600.0

        # === Definir dominio ===
        X, Y = np.meshgrid(self.grid_x, self.grid_y)

        # 🔹 Centro del vórtice (moviéndose lentamente)
        x_c = 0.3 * np.sin(t_h / 2)
        y_c = 0.3 * np.sin(t_h / 3)

        # === Calcular distancia radial (km) ===
        dx = X - x_c
        dy = Y - y_c
        r = np.sqrt(dx**2 + dy**2) + 1e-6

        # === Perfil Rankine ===
        Rc = 0.5  # radio del núcleo (km)
        Vmax_peak = 35.0
        growth = np.clip(t_h / 5.0, 0, 1)
        Vmax = Vmax_peak * growth
        V = np.where(r < Rc, Vmax * (r / Rc), Vmax * (Rc / r))

        # === Rotación ciclónica ===
        u = -dy / r * V
        v =  dx / r * V

        # === Magnitud + ruido ===
        wind_field = np.sqrt(u**2 + v**2)
        wind_field += np.random.normal(0, 1.0, wind_field.shape)
        wind_field = np.clip(wind_field, 0, None)

        self.current = {'wind_speed': wind_field}
        self.wind_field = wind_field

        return time + self.time_resolution

    def get_data(self, outputs):
        if self.current is None:
            return {self.eid: {
                'wind_speed': [],
                'grid_x': [],
                'grid_y': [],
                'wind_shape': [0, 0],
            }}

        return {self.eid: {
            'wind_speed': self.current['wind_speed'].tolist(),
            'grid_x': self.grid_x.tolist(),
            'grid_y': self.grid_y.tolist(),
            'wind_shape': list(self.current['wind_speed'].shape),
        }}
