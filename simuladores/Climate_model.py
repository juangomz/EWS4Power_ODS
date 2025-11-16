import mosaik_api
import numpy as np
import rasterio
from rasterio.enums import Resampling

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'ClimateModel': {
            'public': True,
            'params': [
                'nx', 'ny',
                'x_min', 'x_max', 'y_min', 'y_max',
                'raster_dt_min',
                'rasters'   # diccionario con rutas a cada variable
            ],
            'attrs': [
                'rain_rate', 'gust_speed', 'flash_density',
                'accum_1h', 'accum_6h', 'accum_24h', 'accum_event',
                'grid_x', 'grid_y', 'shape'
            ],
        }
    }
}

RASTERS = {
    'gust_speed': './DANA_SCEN/SCEN_DANA_3x3km_hiRes/G_gust10_COG_CROP.tif',  # m/s (rachas)
    'rain_rate': './DANA_SCEN/SCEN_DANA_3x3km_hiRes/R_hourly_COG_CROP.tif',   # mm/h
    'accum_1h':  './DANA_SCEN/SCEN_DANA_3x3km_hiRes/P1H_accum_COG_CROP.tif', # mm
    'accum_6h':  './DANA_SCEN/SCEN_DANA_3x3km_hiRes/P6H_accum_COG_CROP.tif', # mm
    'accum_24h': './DANA_SCEN/SCEN_DANA_3x3km_hiRes/P24H_accum_COG_CROP.tif' # mm
}

class ClimateModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.sid = None
        self.eid = None
        self.time = 0
        self.nx = 30
        self.ny = 30
        self.x_min, self.x_max = -1.5, 1.5
        self.y_min, self.y_max = -1.5, 1.5
        self.grid_x = None
        self.grid_y = None
        self.raster_dt_min = 10
        self.rasters = {}
        self.ds = {}
        self.current = {}

    def init(self, sid, time_resolution=600, **sim_params):
        self.sid = sid
        self.time_resolution = time_resolution
        return META

    def create(self, num, model, nx=30, ny=30,
           x_min=None, x_max=None, y_min=None, y_max=None,
           raster_dt_min=10, rasters=RASTERS):
        self.nx, self.ny = nx, ny
        self.eid = 'ClimateField'

        self.x_min = x_min or self.x_min
        self.x_max = x_max or self.x_max
        self.y_min = y_min or self.y_min
        self.y_max = y_max or self.y_max

        self.grid_x = np.linspace(self.x_min, self.x_max, self.nx)
        self.grid_y = np.linspace(self.y_min, self.y_max, self.ny)

        self.raster_dt_min = raster_dt_min
        self.rasters = rasters or {}
        self.data = {}   # 🔹 aquí guardaremos todo el contenido

        for key, path in self.rasters.items():
            try:
                ds = rasterio.open(path)
                self.ds[key] = ds

                # 🔹 Carga completa en memoria (144, 30, 30)
                self.data[key] = ds.read().astype(np.float32)
                print(f"[INFO] {key}: cargado en memoria {self.data[key].shape}")

            except Exception as e:
                print(f"[WARN] No se pudo abrir {key}: {path} ({e})")
                self.data[key] = np.zeros((1, self.ny, self.nx), dtype=np.float32)

        return [{'eid': self.eid, 'type': model, 'rel': []}]

    def _read_band(self, key, bidx):
        arr3d = self.data.get(key)
        if arr3d is None:
            return np.zeros((self.ny, self.nx), dtype=np.float32)
        idx = min(bidx - 1, arr3d.shape[0] - 1)
        return arr3d[idx]


    def step(self, time, inputs, max_advance):
        self.time = time
        # Índice base (en minutos)
        base_idx = int(round((time // 60) / self.raster_dt_min)) + 1
        idxs = [base_idx + i for i in range(6)]  # 6 bandas (10 min) por hora
        self.current = {}

        for key in self.rasters.keys():
            frames = []
            for i in idxs:
                if i <= self.ds[key].count:
                    frames.append(self._read_band(key, i))
            if frames:
                self.current[key] = np.mean(frames, axis=0).astype(np.float32)
            else:
                self.current[key] = np.zeros((self.ny, self.nx), dtype=np.float32)

        return int(time + self.time_resolution)

    def get_data(self, outputs):
        if not self.current:
            return {self.eid: {k: [] for k in META['models']['ClimateModel']['attrs']}}

        data = {
            'grid_x': self.grid_x.tolist(),
            'grid_y': self.grid_y.tolist(),
            'shape': [self.ny, self.nx],
        }

        # Asigna solo lo que exista en rasters
        data['rain_rate'] = self.current.get('rain_rate', np.zeros((self.ny, self.nx))).tolist()
        data['gust_speed'] = self.current.get('gust_speed', np.zeros((self.ny, self.nx))).tolist()
        data['flash_density'] = self.current.get('flash_density', np.zeros((self.ny, self.nx))).tolist()
        data['accum_1h'] = self.current.get('accum_1h', np.zeros((self.ny, self.nx))).tolist()
        data['accum_6h'] = self.current.get('accum_6h', np.zeros((self.ny, self.nx))).tolist()
        data['accum_24h'] = self.current.get('accum_24h', np.zeros((self.ny, self.nx))).tolist()
        data['accum_event'] = self.current.get('accum_event', np.zeros((self.ny, self.nx))).tolist()

        return {self.eid: data}
