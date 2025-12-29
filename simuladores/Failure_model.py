import mosaik_api
import numpy as np

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'FailureModel': {
            'public': True,
            'params': ['line_positions'],
            'attrs': ['climate', 'fail_prob', 'shape', 'grid_x', 'grid_y'],
        },
    },
}


# class FailureModel(mosaik_api.Simulator):
#     def __init__(self):
#         super().__init__(META)
#         self.fail_prob = {}
#         self.gust_speed = 0
#         self.grid_x = None
#         self.grid_y = None
#         self.gust_field = None
#         self.shape = None
#         self.time = 0

#     def init(self, sid, **sim_params):
#         return META

#     def create(self,num , model, line_positions=None):
#         # Guardar las posiciones de las líneas
#         if line_positions:
#             self.line_positions = line_positions

#         return [{'eid': 'FailureModel', 'type': model, 'rel': []}]  # Solo una entidad para todo el modelo
    
#     # Interpolación bilineal simple
#     def interp2_bilinear(self, x, y):
#         """
#         Obtains value of the map interpolating betwwen existing 
#         values of the map's grid.

#         Args:
#             x: x position
#             y: y position

#         Returns:
#             Value of the map in the position interpolated
#         """
#         gx, gy, f = self.grid_x, self.grid_y, self.gust_field
#         ny, nx = f.shape

#         # limitar coordenadas dentro de la malla
#         x = np.clip(x, gx[0], gx[-1])
#         y = np.clip(y, gy[0], gy[-1])

#         ix = np.searchsorted(gx, x) - 1
#         iy = np.searchsorted(gy, y) - 1
#         ix = np.clip(ix, 0, nx - 2)
#         iy = np.clip(iy, 0, ny - 2)

#         x0, x1 = gx[ix], gx[ix + 1]
#         y0, y1 = gy[iy], gy[iy + 1]

#         tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
#         ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

#         f00 = f[iy, ix]
#         f10 = f[iy, ix + 1]
#         f01 = f[iy + 1, ix]
#         f11 = f[iy + 1, ix + 1]

#         return (f00 * (1 - tx) * (1 - ty) +
#                 f10 * tx * (1 - ty) +
#                 f01 * (1 - tx) * ty +
#                 f11 * tx * ty)

#     # ============================================================
#     # Muestra viento a lo largo de una línea y calcula fallo
#     def compute_gust_failure(self, x0, y0, x1, y1,
#                     v_th_mean=20.0, alpha_mean=0.0006, beta_mean=3.0,
#                     sigma_vth=2, sigma_alpha=0.0002, sigma_beta=0.4):
#         """
#         Modelo de fallo exponencial (Weibull-like) con dispersión en parámetros.
#         Devuelve probabilidad de fallo y velocidad máxima observada en el tramo.

#         Args:
#             x0: x position of bus 0
#             y0: y poosition of bus 0
#             x1: x position of bus 1
#             y1: y position of bus 1
#             v_th_mean: Mean wind damage threshold. Defaults to 18.0.
#             alpha_mean: mean damage threshold. Defaults to 0.0006.
#             beta_mean: _description_. Defaults to 3.0.
#             sigma_vth: _description_. Defaults to 2.
#             sigma_alpha: _description_. Defaults to 0.0002.
#             sigma_beta: _description_. Defaults to 0.4.

#         Returns:
#             Probability of failure of the line, Max velocity suffered by the line
#         """

#         n = 10
#         FT_TO_KM = 0.0003048
#         xs = np.linspace(x0, x1, n) * FT_TO_KM
#         ys = np.linspace(y0, y1, n) * FT_TO_KM
        
#         # Centrar igual que en PandapowerSim
#         if not hasattr(self, "x_center"):
#             # calcular el centro una sola vez a partir de todas las líneas
#             all_x = [lp['x0'] * FT_TO_KM for lp in self.line_positions.values()] + \
#                     [lp['x1'] * FT_TO_KM for lp in self.line_positions.values()]
#             all_y = [lp['y0'] * FT_TO_KM for lp in self.line_positions.values()] + \
#                     [lp['y1'] * FT_TO_KM for lp in self.line_positions.values()]
#             self.x_center = np.mean(all_x)
#             self.y_center = np.mean(all_y)

#         xs -= self.x_center
#         ys -= self.y_center
    
#         vals = [self.interp2_bilinear(x, y) for x, y in zip(xs, ys)]
#         max_v = float(np.max(vals))

#         # Incluir dispesión en parámetros
#         v_th = np.random.normal(v_th_mean, sigma_vth)       # umbral de inicio de daños
#         alpha = np.random.normal(alpha_mean, sigma_alpha)   # escala del crecimiento
#         beta = np.random.normal(beta_mean, sigma_beta)      # curvatura de la fragilidad

#         # Evitar valores negativos o absurdos
#         v_th = max(10, v_th)
#         alpha = max(0.001, alpha)
#         beta = np.clip(beta, 1.5, 5.0)

#         # Calcular probabilidad de fallo
#         v_eff = max_v
#         if v_eff > v_th:
#             P = 1 - np.exp(-alpha * (v_eff - v_th) ** beta)
#         else:
#             P = 0.0

#         return P, max_v

#     def step(self, time, inputs, max_advance):
#         self.time = time
#         self.gust_field = None

#         # Leer entradas desde wind_sim y pypsa_sim
#         for _, vals in inputs.items():
#             if 'gust_speed' in vals:
#                 w = list(vals['gust_speed'].values())[0]
#                 if isinstance(w, list) or isinstance(w, np.ndarray):
#                     self.gust_field = np.array(w)
#                     if self.shape:
#                         self.gust_field = self.gust_field.reshape(self.shape)
#             if 'line_positions' in vals:
#                 self.line_positions = list(vals['line_positions'].values())[0]
#             if 'grid_x' in vals:
#                 self.grid_x = np.array(list(vals['grid_x'].values())[0])
#             if 'grid_y' in vals:
#                 self.grid_y = np.array(list(vals['grid_y'].values())[0])
#             if 'shape' in vals:
#                 self.shape = tuple(list(vals['shape'].values())[0])

#         # Calcular fallos para todas las líneas a la vez
#         if self.gust_field is not None and self.line_positions:
#             for lid, lp in self.line_positions.items():
#                 P, max_v = self.compute_gust_failure(lp['x0'], lp['y0'], lp['x1'], lp['y1'])
#                 self.fail_prob[lid] = P

#         else:
#             print(f"[t={time / 3600:.0f}h] No hay datos de viento o posiciones.")

#         return time + 3600

#     def get_data(self, outputs):
#         data = {}
#         for eid, attrs in outputs.items():
#             if eid == 'FailureModel':
#                 data[eid] = {
#                     'fail_prob': self.fail_prob,
#                     'gust_speed': self.gust_speed,
#                 }
#         return data

class FailureModel(mosaik_api.Simulator):

    def __init__(self):
        super().__init__(META)

        self.fail_prob = {}          # dict[k][lid]
        self.gust_fields = {}        # dict[k] -> gust field

        self.grid_x = None
        self.grid_y = None
        self.shape = None
        self.line_positions = None

        self.time = 0

    # --------------------------------------------------
    # mosaik API
    # --------------------------------------------------

    def init(self, sid, **sim_params):
        self.sid = sid
        return META

    def create(self, num, model, line_positions=None):
        if line_positions is not None:
            self.line_positions = line_positions

        return [{'eid': 'FailureModel', 'type': model, 'rel': []}]

    # --------------------------------------------------
    # Interpolación bilineal
    # --------------------------------------------------

    def interp2_bilinear(self, x, y):
        gx, gy, f = self.grid_x, self.grid_y, self.gust_field
        ny, nx = f.shape

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

    # --------------------------------------------------
    # Modelo de fallo por ráfaga
    # --------------------------------------------------

    def compute_gust_failure(
        self, x0, y0, x1, y1,
        v_th_mean=20.0, alpha_mean=0.0006, beta_mean=3.0,
        sigma_vth=2.0, sigma_alpha=0.0002, sigma_beta=0.4
    ):
        n = 10
        FT_TO_KM = 0.0003048

        xs = np.linspace(x0, x1, n) * FT_TO_KM
        ys = np.linspace(y0, y1, n) * FT_TO_KM

        if not hasattr(self, "x_center"):
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

        v_th = max(10.0, np.random.normal(v_th_mean, sigma_vth))
        alpha = max(0.001, np.random.normal(alpha_mean, sigma_alpha))
        beta = np.clip(np.random.normal(beta_mean, sigma_beta), 1.5, 5.0)

        if max_v > v_th:
            P = 1.0 - np.exp(-alpha * (max_v - v_th) ** beta)
        else:
            P = 0.0

        return float(P), max_v

    # --------------------------------------------------
    # step
    # --------------------------------------------------

    def step(self, time, inputs, max_advance):
        self.time = time

        self.gust_fields = {}
        self.fail_prob = {}

        # Leer entradas
        for _, vals in inputs.items():

            if 'grid_x' in vals:
                self.grid_x = np.array(list(vals['grid_x'].values())[0])

            if 'grid_y' in vals:
                self.grid_y = np.array(list(vals['grid_y'].values())[0])

            if 'shape' in vals:
                self.shape = tuple(list(vals['shape'].values())[0])

            if 'line_positions' in vals:
                self.line_positions = list(vals['line_positions'].values())[0]

            if 'climate' in vals:
                climate = list(vals['climate'].values())[0]  # dict {k: field}

                for k, field in climate.items():
                    gust = np.array(field['gust_speed'])
                    if self.shape:
                        gust = gust.reshape(self.shape)
                    self.gust_fields[int(k)] = gust

        # Calcular fallos por k
        if self.gust_fields and self.line_positions:
            for k, gust_field in self.gust_fields.items():
                self.gust_field = gust_field
                self.fail_prob[k] = {}

                for lid, lp in self.line_positions.items():
                    P, _ = self.compute_gust_failure(
                        lp['x0'], lp['y0'], lp['x1'], lp['y1']
                    )
                    self.fail_prob[k][lid] = P
        else:
            print(f"[t={time / 3600:.0f}h] FailureModel: datos incompletos.")

        return time + 3600

    # --------------------------------------------------
    # get_data
    # --------------------------------------------------

    def get_data(self, outputs):
        return {
            'FailureModel': {
                'fail_prob': self.fail_prob
            }
        }
