import mosaik_api
import numpy as np

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'WindSim2D': {
            'public': True,
            'params': ['nx', 'ny'],  # grid size
            'attrs': ['wind_speed', 'grid_x', 'grid_y', 'wind_shape'],  # output attribute
        }
    }
}


class WindSim2D(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.sid = None
        self.eid = None
        self.time = 0
        self.nx = 10
        self.ny = 10
        self.current = None
        self.grid_x = None
        self.gird_y = None
        
        

    def init(self, sid, time_resolution=3600, **sim_params):
        self.sid = sid
        self.time_resolution = time_resolution
        return META

    def create(self, num, model, nx=10, ny=10):
        self.nx = nx
        self.ny = ny
        self.eid = 'WindField'
        
        # ✅ Crear una malla espacial base
        self.grid_x = np.linspace(0, 10, self.nx)
        self.grid_y = np.linspace(0, 10, self.ny)
        
        return [{'eid': self.eid, 'type': model, 'rel': []}]

    def step(self, time, inputs, max_advance):
        self.time = time

        # --- Temporal pattern ---
        base_wind = 10 + 5 * np.sin(2 * np.pi * time / (24 * 3600))

        # --- Spatial variation ---
        X, Y = np.meshgrid(self.grid_x, self.grid_y)

        # Base pattern + random noise
        wind_field = base_wind + 5 * np.sin(X + Y + time / 3600) + np.random.normal(0, 0.5, (self.ny, self.nx))

        # Store the current 2D field
        self.current = {'wind_speed': wind_field}

        return time + self.time_resolution

    def get_data(self, outputs):
        # wind_field es un np.ndarray (ny, nx)
        if isinstance(self.current['wind_speed'], np.ndarray):
            wind_data = self.current['wind_speed'].tolist()  # ✅ convertir a lista
        else:
            wind_data = self.current['wind_speed']

        return {self.eid: {
            'wind_speed': self.current['wind_speed'].tolist(),  # ✅ importante
            'grid_x': self.grid_x.tolist(),
            'grid_y': self.grid_y.tolist(),
            'wind_shape': list(self.current['wind_speed'].shape),
        }}
