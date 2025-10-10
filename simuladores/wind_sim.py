import mosaik_api
import numpy as np

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'WindSim': {
            'public': True,
            'params': [],
            'attrs': ['wind_speed'],
        }
    }
}

class WindSim(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid = None
        self.time = 0

    def init(self, sid, **sim_params):
        return META

    def create(self, num, model):
        self.eid = 'WindSensor'
        return [{'eid': self.eid, 'type': model, 'rel': []}]

    def step(self, time, inputs, max_advance):
        self.time = time
        wind_speed = 8 + 5 * np.sin(2 * np.pi * time / (24 * 3600))
        self.current = {'wind_speed': wind_speed}
        return time + 3600

    def get_data(self, outputs):
        return {self.eid: self.current}