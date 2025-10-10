import mosaik_api
from simuladores.logger import Logger

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'FailureModel': {
            'public': True,
            'params': [],
            'attrs': ['wind_speed', 'line_status'],  # ✅ solo strings
        },
    },
}

class FailureModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.entities = {}  # Guardar estado de cada entidad
        self.wind_speed = 0

    def init(self, sid, **sim_params):
        return META

    def create(self, num, model):
        """Crea tantas entidades como líneas haya en la red."""
        entities = []
        for i in range(num):
            eid = f'FailureProc_{i}'
            entities.append({'eid': eid, 'type': model, 'rel': []})
            # Inicializar estado de cada entidad
            self.entities[eid] = {'line_status': 1, 'wind_speed': 0}
        return entities

    def step(self, time, inputs, max_advance):
        """Actualiza el estado de cada entidad según el viento."""
        wind_speed = 0
        if inputs:
            # Obtener el viento de la fuente conectada
            src = list(inputs.keys())[0]
            vals = inputs[src]
            if 'wind_speed' in vals:
                wind_speed = list(vals['wind_speed'].values())[0]

        self.wind_speed = wind_speed

        for eid, state in self.entities.items():
            if state['line_status'] != 0:
                state['line_status'] = 0 if wind_speed > 12 else 1
            state['wind_speed'] = wind_speed

            print(f"[t={time/3600:.0f}h] {eid}: wind={wind_speed:.1f} → line={state['line_status']}")

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

