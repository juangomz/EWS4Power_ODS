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
        """
        Obtains value of the map interpolating betwwen existing 
        values of the map's grid.

        Args:
            x: x position
            y: y position

        Returns:
            Value of the map in the position interpolated
        """
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
    def compute_wind_failure(self, x0, y0, x1, y1,
                    v_th_mean=18.0, alpha_mean=0.0006, beta_mean=3.0,
                    sigma_vth=2, sigma_alpha=0.0002, sigma_beta=0.4):
        """
        Modelo de fallo exponencial (Weibull-like) con dispersión en parámetros.
        Devuelve probabilidad de fallo y velocidad máxima observada en el tramo.

        Args:
            x0: x position of bus 0
            y0: y poosition of bus 0
            x1: x position of bus 1
            y1: y position of bus 1
            v_th_mean: Mean wind damage threshold. Defaults to 18.0.
            alpha_mean: mean damage threshold. Defaults to 0.0006.
            beta_mean: _description_. Defaults to 3.0.
            sigma_vth: _description_. Defaults to 2.
            sigma_alpha: _description_. Defaults to 0.0002.
            sigma_beta: _description_. Defaults to 0.4.

        Returns:
            Probability of failure of the line, Max velocity suffered by the line
        """

        n = 10
        FT_TO_KM = 0.0003048
        xs = np.linspace(x0, x1, n) * FT_TO_KM
        ys = np.linspace(y0, y1, n) * FT_TO_KM
        
        # Centrar igual que en PandapowerSim
        if not hasattr(self, "x_center"):
            # calcular el centro una sola vez a partir de todas las líneas
            all_x = [lp['x0'] * FT_TO_KM for lp in self.line_positions.values()] + \
                    [lp['x1'] * FT_TO_KM for lp in self.line_positions.values()]
            all_y = [lp['y0'] * FT_TO_KM for lp in self.line_positions.values()] + \
                    [lp['y1'] * FT_TO_KM for lp in self.line_positions.values()]
            self.x_center = np.mean(all_x)
            self.y_center = np.mean(all_y)

        xs -= self.x_center
        ys -= self.y_center
    
        vals = [self.interp2_bilinear(x, y) for x, y in zip(xs, ys)]
        max_v = float(np.max(vals))

        # Incluir dispesión en parámetros
        v_th = np.random.normal(v_th_mean, sigma_vth)       # umbral de inicio de daños
        alpha = np.random.normal(alpha_mean, sigma_alpha)   # escala del crecimiento
        beta = np.random.normal(beta_mean, sigma_beta)      # curvatura de la fragilidad

        # Evitar valores negativos o absurdos
        v_th = max(10, v_th)
        alpha = max(0.001, alpha)
        beta = np.clip(beta, 1.5, 5.0)

        # Calcular probabilidad de fallo
        v_eff = max_v
        if v_eff > v_th:
            P = 1 - np.exp(-alpha * (v_eff - v_th) ** beta)
        else:
            P = 0.0

        return P, max_v
    
     # ============================================================
    def step(self, time, inputs, max_advance):
        self.time = time
        self.wind_field = None

        # leer entradas desde wind_sim y pypsa_sim
        for _, vals in inputs.items():
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

        # Calcular fallos para cada línea
        if self.wind_field is not None and self.line_positions:
            for eid, state in self.entities.items():
                # Extraer posición asociada a esta línea
                i = int(eid.split('_')[-1])
                lids = list(self.line_positions.keys())
                if i < len(lids):
                    lid = lids[i]
                    lp = self.line_positions[lid]
                    P, max_v = self.compute_failure(lp['x0'], lp['y0'], lp['x1'], lp['y1'])
                    if np.random.rand() < P:
                        status = 0  # fallo
                    else:
                        status = 1  # operativo
                    state['line_status'] = status
                    state['wind_speed'] = max_v
                    # print(f"[t={time/3600:.0f}h] {lid}: mean_v={mean_v:.1f} → line={status}")
        else:
            print(f"[t={time/3600:.0f}h] No hay datos de viento o posiciones.")

        return time + 3600

    # Entrega de datos a Mosaik
    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            if eid in self.entities:
                # devolvemos siempre los dos atributos
                data[eid] = {
                    'line_status': self.entities[eid]['line_status'],
                    'wind_speed': self.entities[eid]['wind_speed'],
                }

        return data

