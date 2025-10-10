import mosaik_api
from simuladores.logger import Logger
import numpy as np

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'FailureModel': {
            'public': True,
            'params': ['line_positions'],  # ✅ ahora sí lo acepta
            'attrs': ['wind_speed', 'line_status', 'wind_shape', 'grid_x', 'grid_y'],
        },
    },
}


class FailureModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.entities = {}  # Guardar estado de cada entidad
        self.wind_speed = 0
        self.grid_x = None
        self.grid_y = None
        self.wind_field = None
        self.wind_shape = None
        self.time = 0

    def init(self, sid, **sim_params):
        return META

    def create(self, num, model, line_positions=None):
        entities = []
        for i in range(num):
            eid = f'FailureProc_{i}'
            entities.append({'eid': eid, 'type': model, 'rel': []})
            self.entities[eid] = {
                'line_status': 1,
                'wind_speed': 0,
            }

        # ✅ Guardar las posiciones una sola vez
        if line_positions:
            self.line_positions = line_positions

        return entities

    # ============================================================
    # Interpolación bilineal simple
    def interp2_bilinear(self, x, y):
        gx, gy, f = self.grid_x, self.grid_y, self.wind_field
        ny, nx = f.shape

        # limitar coordenadas dentro de la malla
        x = np.clip(x, gx[0], gx[-1])
        y = np.clip(y, gy[0], gy[-1])

        ix = np.searchsorted(gx, x) - 1
        iy = np.searchsorted(gy, y) - 1
        ix = np.clip(ix, 0, nx - 2)
        iy = np.clip(iy, 0, ny - 2)

        x0, x1 = gx[ix], gx[ix + 1]
        y0, y1 = gy[iy], gy[iy + 1]

        tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

        f00 = f[iy, ix]
        f10 = f[iy, ix + 1]
        f01 = f[iy + 1, ix]
        f11 = f[iy + 1, ix + 1]

        return (f00 * (1 - tx) * (1 - ty) +
                f10 * tx * (1 - ty) +
                f01 * (1 - tx) * ty +
                f11 * tx * ty)

    # ============================================================
    # Muestra viento a lo largo de una línea y calcula fallo
    def compute_failure(self, x0, y0, x1, y1, threshold=10.0):
        n = 10
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        vals = [self.interp2_bilinear(x, y) for x, y in zip(xs, ys)]
        mean_v = float(np.mean(vals))
        # Regla simple: si el promedio > umbral, línea falla
        status = 0 if mean_v > threshold else 1
        return status, mean_v
    
     # ============================================================
    def step(self, time, inputs, max_advance):
        self.time = time
        self.wind_field = None

        # 1️⃣ Leer entradas desde WindSim y PyPSASim
        for src, vals in inputs.items():
            if 'wind_speed' in vals:
                w = list(vals['wind_speed'].values())[0]
                if isinstance(w, list) or isinstance(w, np.ndarray):
                    self.wind_field = np.array(w)
                    if self.wind_shape:
                        self.wind_field = self.wind_field.reshape(self.wind_shape)
            if 'line_positions' in vals:
                self.line_positions = list(vals['line_positions'].values())[0]
            if 'grid_x' in vals:
                self.grid_x = np.array(list(vals['grid_x'].values())[0])
            if 'grid_y' in vals:
                self.grid_y = np.array(list(vals['grid_y'].values())[0])
            if 'wind_shape' in vals:
                self.wind_shape = tuple(list(vals['wind_shape'].values())[0])

        # 2️⃣ Calcular fallos si hay datos suficientes
        if self.wind_field is not None and self.line_positions:
            for eid, state in self.entities.items():
                # Extraer posición asociada a esta línea
                i = int(eid.split('_')[-1])
                lids = list(self.line_positions.keys())
                if i < len(lids):
                    lid = lids[i]
                    lp = self.line_positions[lid]
                    status, mean_v = self.compute_failure(lp['x0'], lp['y0'], lp['x1'], lp['y1'])
                    state['line_status'] = status
                    state['wind_speed'] = mean_v
                    print(f"[t={time/3600:.0f}h] {lid}: mean_v={mean_v:.1f} → line={status}")
        else:
            print(f"[t={time/3600:.0f}h] ⚠️ No hay datos de viento o posiciones.")

        return time + 3600

    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            if eid in self.entities:
                # devolvemos siempre los dos atributos
                data[eid] = {
                    'line_status': self.entities[eid]['line_status'],
                    'wind_speed': self.entities[eid]['wind_speed'],
                }
        print("📤 get_data() →", data)
        return data

