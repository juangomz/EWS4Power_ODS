import mosaik_api
import numpy as np

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'ClimateModel': {
            'public': True,
            'params': ['nx', 'ny', 'x_min', 'x_max', 'y_min', 'y_max'],
            'attrs': [
                'rain_rate', 'gust_speed', 'flash_density',
                'grid_x', 'grid_y', 'shape'
            ],
        }
    }
}

class ClimateModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)

        # Grid
        self.nx = 30
        self.ny = 30
        self.x_min, self.x_max = -1.5, 1.5
        self.y_min, self.y_max = -1.5, 1.5

        self.grid_x = None
        self.grid_y = None
        self.q2 = None
        self.r2 = None

        self.time_resolution = 600
        self.current = {}

    def init(self, sid, time_resolution=600, **sim_params):
        self.sid = sid
        self.time_resolution = time_resolution
        return META

    def create(self, num, model, nx=30, ny=30,
               x_min=None, x_max=None, y_min=None, y_max=None):

        self.nx, self.ny = nx, ny
        self.eid = 'ClimateField'

        # Bounds
        self.x_min = x_min or self.x_min
        self.x_max = x_max or self.x_max
        self.y_min = y_min or self.y_min
        self.y_max = y_max or self.y_max

        # Grid
        self.grid_x = np.linspace(self.x_min, self.x_max, self.nx)
        self.grid_y = np.linspace(self.y_min, self.y_max, self.ny)

        # Normalized 0–1
        q = (self.grid_x - self.x_min) / (self.x_max - self.x_min)
        r = (self.grid_y - self.y_min) / (self.y_max - self.y_min)

        # 2D mesh
        self.q2, self.r2 = np.meshgrid(q, r)

        return [{'eid': self.eid, 'type': model, 'rel': []}]

    def step(self, time, inputs, max_advance):
        # time in hours
        t = time / 3600.0

        # temporal frequency (real movement)
        omega = 0.5  # prueba 0.5, 1, 2 para ver velocidades mayores

        phase = 2 * np.pi

        # add time into the wave directly (non-integer shift)
        arg = phase * self.q2 + omega * t

        # synthetic fields with real time variation
        w1 = 20 + 5 * np.sin(arg) * np.cos(phase * self.r2 + omega * t)
        w2 = 22 + 3 * np.cos(arg) * np.sin(phase * self.r2 + omega * t)
        w3 = 0.5 + 0.1 * np.sin(arg) * np.cos(phase * self.r2 + omega * t)

        self.current['rain_rate'] = w1
        self.current['gust_speed'] = w2
        self.current['flash_density'] = w3

        return int(time + self.time_resolution)

    def get_data(self, outputs):
        return {
            self.eid: {
                'grid_x': self.grid_x.tolist(),
                'grid_y': self.grid_y.tolist(),
                'shape': [self.ny, self.nx],
                'rain_rate': self.current['rain_rate'].tolist(),
                'gust_speed': self.current['gust_speed'].tolist(),
                'flash_density': self.current['flash_density'].tolist(),
            }
        }
