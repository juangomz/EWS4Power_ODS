import mosaik_api
from collections import deque  # para el BFS de downstream
from algorithms.optimize_switches_gurobi import optimize_switches_gurobi
from algorithms.optimize_switches_ga import optimize_switches_ga

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'OpDecisionModel': {
            'public': True,
            'attrs': ['fail_prob',
                      'repair_plan',
                      'line_status',
                      'lines',
                      'buses',
                      'switches',
                      'transformers',
                      'switch_plan'
            ],
        },
    },
}


class OpDecisionModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)

        # Estado interno
        self.line_status = {}        # Estado actual de las líneas {lid: 0/1}
        self.fail_prob = {}          # Probabilidad de fallo recibida {lid: p}
        self.failed_lines = set()    # Líneas caídas acumuladas
        self.repair_time = 3600      # Tiempo mínimo para reparar (1h)
        self.resources = 2           # Cuadrillas
        self.ongoing_repairs = {}    # {line_id: finish_time}
        self.current_repair_plan = {}  # Se envía al grid
        self.switch_plan = {}        # Plan de como ajustar los switches para minimzar ENS
        self.prev_switch_state = {} 

        self.lines = None            # DataFrame de líneas (pandapower)
        self.buses = None            # DataFrame de buses
        self.switches = None         # DataFrame o dict de switches bus-bus
        self.switches_buses = None   # from-bus a to-bus. El modelo no le gusta recibir con lineas
        self.transformers = None
        
        

        # Para downstream
        self.graph = {}              # grafo from_bus -> [to_bus,...]
        self.initialized = False     # para calcular downstream una sola vez

    def init(self, sid, **sim_params):
        return META

    def create(self, num, model):
        return [{'eid': 'OpDecisionModel', 'type': model}]
    
    # ===================
    # CÁLCULO DOWNSTREAM
    # ===================

    def _build_graph(self):
        """
        Construye un grafo dirigido básico from_bus -> to_bus
        usando el DataFrame self.lines (pandapower).
        """
        self.graph = {}
        if self.lines is None:
            return

        for lid in self.lines.index:
            fb = self.lines.at[lid, 'from_bus']
            tb = self.lines.at[lid, 'to_bus']
            if fb not in self.graph:
                self.graph[fb] = []
            self.graph[fb].append(tb)

    def _compute_downstream_buses(self):
        """
        Para cada línea (fila en self.lines), calcula cuántos buses
        hay aguas abajo del to_bus en el grafo base y lo guarda
        en la columna 'downstream_buses' del DataFrame.
        """
        if self.lines is None:
            return

        # creamos la columna si no existe
        if 'downstream_buses' not in self.lines.columns:
            self.lines['downstream_buses'] = 0

        for lid in self.lines.index:
            tb = self.lines.at[lid, 'to_bus']

            visited = set([tb])
            queue = deque([tb])

            while queue:
                b = queue.popleft()
                for child in self.graph.get(b, []):
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)

            # número de buses aguas abajo (incluyendo el to_bus)
            self.lines.at[lid, 'downstream_buses'] = len(visited)
        
    # ==============
    # SIM LOOP
    # ==============

    def step(self, time, inputs, max_advance):
        """Recibe fail_prob, line_status_prev, topología y decide reparaciones + switches."""

        # Leer inputs desde otros simuladores
        for src, vals in inputs.items():
            if 'fail_prob' in vals:
                self.fail_prob = list(vals['fail_prob'].values())[0]
            if 'line_status' in vals:
                self.line_status = list(vals['line_status'].values())[0]
            if 'lines' in vals:
                self.lines = list(vals['lines'].values())[0]   # aquí llega el DataFrame
            if 'buses' in vals:
                self.buses = list(vals['buses'].values())[0]
            if 'switches' in vals:
                self.switches = list(vals['switches'].values())[0]
            if 'transformers' in vals:
                self.transformers = list(vals['transformers'].values())[0]

        # Inicializar grafo + downstream una sola vez (cuando ya tenemos líneas)
        if not self.initialized and self.lines is not None:
            self._build_graph()
            self._compute_downstream_buses()
            self.initialized = True

        # === Detectar nuevas líneas caídas ===
        for lid, status in self.line_status.items():
            if status == 0:  # caída
                self.failed_lines.add(lid)
                
        # (De momento solo calculamos el plan, si luego quieres
        #  aplicar el estado de los switches al grid, lo mandarás
        #  como nueva salida/atributo.)

        # === Finalizar reparaciones que terminan ahora ===
        to_remove = []
        for lid, finish_time in self.ongoing_repairs.items():
            if time >= finish_time:
                self.line_status[lid] = 1
                self.lines.at[lid, "in_service"] = True
                to_remove.append(lid)

        for lid in to_remove:
            del self.ongoing_repairs[lid]
            if lid in self.failed_lines:
                self.failed_lines.remove(lid)

        # === Asignar cuadrillas libres a fallos pendientes ===
        free_crews = self.resources - len(self.ongoing_repairs)
        if free_crews > 0:
            for lid in sorted(self.failed_lines):
                if free_crews <= 0:
                    break
                finish = time + self.repair_time
                self.ongoing_repairs[lid] = finish
                free_crews -= 1

        # === Generar repair_plan para grid ===
        self.current_repair_plan = {
            lid: {'finish_time': finish}
            for lid, finish in self.ongoing_repairs.items()
        }
                
        self.switches_buses = self.switches.copy()
        for sid in self.switches.index:
            self.switches_buses.at[sid, "bus"] = self.lines.at[self.switches.at[sid,"element"], "from_bus"]
            self.switches_buses.at[sid, "element"] = self.lines.at[self.switches.at[sid,"element"], "to_bus"]
        # === LIMPIAR LINEAS FIJAS QUE SON SWITCHES
        self.lines = self.lines.drop([12, 13, 14], errors="ignore")

        # === Correr optimización de switches ===
        # self.switch_plan = optimize_switches_gurobi(self.buses, self.lines, self.switches_buses, self.transformers, self.fail_prob)
        self.switch_plan = optimize_switches_ga(self.buses, self.lines, self.switches_buses, self.transformers, self.fail_prob)

        return time + 3600  # paso 1 hora

    def get_data(self, outputs):
        return {
            'OpDecisionModel': {
                'repair_plan': self.current_repair_plan,
                'line_status': self.line_status,
                'switch_plan': self.switch_plan
            }
        }
