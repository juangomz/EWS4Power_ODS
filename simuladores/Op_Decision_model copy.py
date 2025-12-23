import mosaik_api
import gurobipy as gp
from collections import deque  # para el BFS de downstream

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

    # ======================
    # OPTIMIZACIÓN SWITCHES
    # ======================

    def optimize_switches(self):
        """Optimiza la configuración de switches con:
        - radialidad estricta,
        - riesgo por probabilidad de fallo,
        - penalización por operaciones (gamma),
        - continuidad temporal del estado previo.
        """

        model = gp.Model("switch_opt")
        model.Params.OutputFlag = 0

        # ===== PARÁMETROS =====
        ALPHA = 1000.0       # peso del riesgo (probabilidad)
        SWITCH_COST = 1      # coste de cambiar el estado (gamma)
        STATUS_COST = 0.1    # coste pequeño por tener un switch cerrado (para evitar cerrados inútiles)

        # ===== VARIABLES =====

        # Energización de buses
        e = {b: model.addVar(vtype=gp.GRB.BINARY, name=f"e_{b}")
            for b in self.buses.index}

        # Switch cerrado/abierto
        x = {sid: model.addVar(vtype=gp.GRB.BINARY, name=f"x_sw_{sid}")
            for sid in self.switches.index}

        # Línea energizando
        y_line = {lid: model.addVar(vtype=gp.GRB.BINARY, name=f"y_line_{lid}")
                for lid in self.lines.index}

        # Switch energizando
        y_sw = {sid: model.addVar(vtype=gp.GRB.BINARY, name=f"y_sw_{sid}")
                for sid in self.switches.index}

        # Cambio de estado respecto al tiempo anterior
        gamma = {sid: model.addVar(vtype=gp.GRB.BINARY, name=f"gamma_{sid}")
                for sid in self.switches.index}

        # ===== SLACK =====

        slack_candidates = self.buses[self.buses["name"] == "150"]
        if len(slack_candidates) == 0:
            raise RuntimeError("Slack bus (name=='150') no encontrado.")
        slack = slack_candidates.index[0]

        model.addConstr(e[slack] == 1, name="slack_energizado")

        # ===== PROPAGACIÓN DE ENERGÍA =====

        # Líneas fijas
        for lid in self.lines.index:
            fb = int(self.lines.at[lid, "from_bus"])
            tb = int(self.lines.at[lid, "to_bus"])
            status = int(self.line_status.get(lid, 1))

            if status == 1:
                model.addConstr(e[tb] <= e[fb], name=f"line_prop1_{lid}")
                model.addConstr(e[fb] <= e[tb], name=f"line_prop2_{lid}")

        # Switches (bus-element)
        for sid in self.switches.index:
            a = int(self.switches.at[sid, "bus"])
            b = int(self.switches.at[sid, "element"])

            # Propagación: si x=1 los buses pueden compartir energía
            model.addConstr(e[b] <= e[a] + (1 - x[sid]), name=f"sw_prop1_{sid}")
            model.addConstr(e[a] <= e[b] + (1 - x[sid]), name=f"sw_prop2_{sid}")

            # y_sw solo si switch está cerrado y ambos buses energizados
            model.addConstr(y_sw[sid] <= x[sid], name=f"y_le_x_{sid}")
            model.addConstr(y_sw[sid] <= e[a], name=f"ysw_a_{sid}")
            model.addConstr(y_sw[sid] <= e[b], name=f"ysw_b_{sid}")

        # ===== y_line lógico =====

        for lid in self.lines.index:
            fb = int(self.lines.at[lid, "from_bus"])
            tb = int(self.lines.at[lid, "to_bus"])
            status = int(self.line_status.get(lid, 1))

            model.addConstr(y_line[lid] <= status)
            model.addConstr(y_line[lid] <= e[fb])
            model.addConstr(y_line[lid] <= e[tb])

        # ===== RADIALIDAD =====
        model.addConstr(
            gp.quicksum(y_line.values()) + gp.quicksum(y_sw.values())
            == gp.quicksum(e.values()) - 1,
            name="radialidad"
        )

        # ===== CONTINUIDAD TEMPORAL DE SWITCHES =====

        for sid in self.switches.index:
            z_prev = int(self.prev_switch_state.get(sid, self.switches.at[sid, "closed"]))
            model.addConstr(gamma[sid] >= x[sid] - z_prev, name=f"gamma_pos_{sid}")
            model.addConstr(gamma[sid] >= z_prev - x[sid], name=f"gamma_neg_{sid}")

        # ===== OBJETIVO =====

        # ENS
        ens_term = gp.quicksum(1 - e[b] for b in e)

        # Riesgo
        risk_term = gp.quicksum(
            ALPHA * float(self.fail_prob.get(lid, 0.0)) * y_line[lid]
            for lid in self.lines.index
        )

        # Coste de operación
        op_term = SWITCH_COST * gp.quicksum(gamma[sid] for sid in gamma)

        # Coste por tener switches cerrados (para abrir los que no aportan)
        status_term = STATUS_COST * gp.quicksum(x[sid] for sid in x)

        model.setObjective(ens_term + risk_term + op_term + status_term,
                        gp.GRB.MINIMIZE)

        model.optimize()

        return {sid: int(round(x[sid].X)) for sid in x}



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

        # Inicializar grafo + downstream una sola vez (cuando ya tenemos líneas)
        if not self.initialized and self.lines is not None:
            self._build_graph()
            self._compute_downstream_buses()
            self.initialized = True

        # === Detectar nuevas líneas caídas ===
        for lid, status in self.line_status.items():
            if status == 0:  # caída
                self.failed_lines.add(lid)

        # === Correr optimización de switches ===
        self.switch_plan = self.optimize_switches()

        # (De momento solo calculamos el plan, si luego quieres
        #  aplicar el estado de los switches al grid, lo mandarás
        #  como nueva salida/atributo.)

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

        return time + 3600  # paso 1 hora

    def get_data(self, outputs):
        return {
            'OpDecisionModel': {
                'repair_plan': self.current_repair_plan,
                'line_status': self.line_status,
                'switch_plan': self.switch_plan
            }
        }