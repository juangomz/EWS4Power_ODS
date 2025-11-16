import mosaik_api
import gurobipy as gp

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'OpDecisionModel': {
            'public': True,
            'attrs': ['fail_prob', 'repair_plan', 'line_status'],
        },
    },
}


class OpDecisionModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)

        # Estado interno
        self.line_status = {}        # Estado actual de las líneas
        self.fail_prob = {}          # Probabilidad de fallo recibida
        self.failed_lines = set()    # Líneas caídas acumuladas
        self.repair_time = 3600      # Tiempo mínimo para reparar (1h)
        self.resources = 2           # Cuadrillas
        self.ongoing_repairs = {}    # {line_id: finish_time}
        self.current_repair_plan = {}  # Se envía al grid

    def init(self, sid, **sim_params):
        return META

    def create(self, num, model):
        return [{'eid': 'OpDecisionModel', 'type': model}]

    def step(self, time, inputs, max_advance):
        """Recibe fail_prob y line_status_prev, decide reparaciones."""
        # Leer inputs
        for src, vals in inputs.items():
            if 'fail_prob' in vals:
                self.fail_prob = list(vals['fail_prob'].values())[0]
            if 'line_status' in vals:
                self.line_status = list(vals['line_status'].values())[0]

        # === Detectar nuevas líneas caídas ===
        for lid, status in self.line_status.items():
            if status == 0:  # caída
                self.failed_lines.add(lid)

        # === Finalizar reparaciones que terminan ahora ===
        to_remove = []
        for lid, finish_time in self.ongoing_repairs.items():
            if time >= finish_time:
                self.line_status[lid] = 1
                to_remove.append(lid)

        for lid in to_remove:
            del self.ongoing_repairs[lid]
            if lid in self.failed_lines:
                self.failed_lines.remove(lid)

        # === Asignar cuadrillas libres a fallos pendientes ===
        free_crews = self.resources - len(self.ongoing_repairs)
        new_repairs = []

        if free_crews > 0:
            for lid in sorted(self.failed_lines):
                if free_crews <= 0:
                    break
                finish = time + self.repair_time
                self.ongoing_repairs[lid] = finish
                new_repairs.append((lid, finish))
                free_crews -= 1

        # === Generar repair_plan para grid ===
        self.current_repair_plan = {
            lid: {'finish_time': finish}
            for lid, finish in self.ongoing_repairs.items()
        }

        return time + 3600  # paso 1 hora

    def get_data(self, outputs):
        return {
            'OpDecisionModel': {
                'repair_plan': self.current_repair_plan,
                'line_status': self.line_status
            }
        }
