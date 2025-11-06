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
        self.entities = {}
        self.failed_lines = {}
        self.repair_time = 3 * 3600  # 3 horas por reparación
        self.last_repair_time = {}

    def init(self, sid, **sim_params):
        return META

    def create(self, num, model):
        entities = []
        for i in range(num):
            eid = f'Repair_{i}'
            entities.append({'eid': eid, 'type': model, 'rel': []})
            self.entities[eid] = {'repaired_lines': []}
        return entities

    def step(self, time, inputs, max_advance):
        repaired = []

        for _, vals in inputs.items():
            if 'line_status' in vals:
                status_dict = list(vals['line_status'].values())[0]
                for lid, status in status_dict.items():
                    if status == 0:
                        # línea fallada
                        if lid not in self.failed_lines:
                            self.failed_lines[lid] = time
                    else:
                        # si ya está operativa, quitar de la lista de fallos
                        self.failed_lines.pop(lid, None)

        # simular reparaciones después de cierto tiempo
        for lid, fail_time in list(self.failed_lines.items()):
            if time - fail_time >= self.repair_time:
                repaired.append(lid)
                del self.failed_lines[lid]

        for eid in self.entities:
            self.entities[eid]['repaired_lines'] = repaired

        return time + 3600

    def get_data(self, outputs):
        data = {}
        for eid, attrs in outputs.items():
            if 'repaired_lines' in attrs:
                data[eid] = {'repaired_lines': self.entities[eid]['repaired_lines']}
        return data
