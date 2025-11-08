import mosaik_api
import numpy as np

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'OpDecisionModel': {
            'public': True,
            'attrs': ['line_status', 'repaired_lines'],
        },
    },
}

class OpDecisionModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.failed_lines = {}   # Mapa de líneas falladas (lid -> tiempo de fallo)
        self.repair_time = 3 * 3600  # Tiempo para reparar una línea (3 horas)
        self.resources = 2        # Cuadrillas disponibles
        self.ongoing_repairs = {} # Mapa de reparaciones en curso (lid -> tiempo de inicio)
        self.repaired_lines = []  # Lista de líneas reparadas en este paso
        self.line_status = {}     # Diccionario global del estado de todas las líneas

    def init(self, sid, **sim_params):
        return META

    def create(self, num, model):
        # Inicializamos line_status con todas las líneas operativas (estado = 1)
        self.line_status = {f'Line_{i}': 1 for i in range(num)}  # Todas las líneas operativas al principio
        return [{'eid': 'OpDecisionModel', 'type': model, 'rel': []}]

    def step(self, time, inputs, max_advance):
        repaired = []  # Lista de líneas reparadas
        line_status_updates = {}  # Diccionario de estados de las líneas

        # Leer entrada de FailureModel (status de las líneas)
        line_status_inputs = {}
        for _, vals in inputs.items():
            if 'line_status' in vals:
                line_status_inputs = vals['line_status']

        # Procesar líneas falladas y asignar recursos de reparación
        for src_id, _ in line_status_inputs.items():
            line_dict = line_status_inputs[src_id]
            for line_id, status in line_dict.items():
                if status == 0:  # La línea está caída
                    if line_id not in self.failed_lines:
                        self.failed_lines[line_id] = time  # Registrar el tiempo de fallo

        # Asignar recursos (reparar líneas)
        while len(self.ongoing_repairs) < self.resources and self.failed_lines:
            lid, fail_time = self.failed_lines.popitem()
            self.ongoing_repairs[lid] = time  # Comienza la reparación de la línea

        # Reparar líneas que han alcanzado el tiempo de reparación
        for lid, start_time in list(self.ongoing_repairs.items()):
            if time - start_time >= self.repair_time:
                repaired.append(lid)
                del self.ongoing_repairs[lid]

        # Actualizar el estado de las líneas
        for lid in repaired:
            line_status_updates[lid] = 1  # La línea se ha reparado, por lo tanto su estado es 1
            self.repaired_lines.append(lid)  # Agregar a las reparadas

        self.line_status = line_dict
        for lid in self.repaired_lines:
            self.line_status[lid] = 1

        return time + 3600

    def get_data(self, outputs=None):
        """Devuelve las líneas reparadas y el estado completo de todas las líneas a Mosaik"""
        return {
            'OpDecisionModel': {
                'repaired_lines': self.repaired_lines,
                'line_status': self.line_status  # Devuelve el estado completo de todas las líneas
            }
        }
