import mosaik_api
import pandapower as pp
import pandas as pd
import numpy as np
import csv
import os
from simuladores.logger import Logger
import json
import numpy as np

from plot_fn import plot_network, plot_R_curve

data_dir = './data/red_electrica'

def net_cigre15_mv(IN_PATH):
    """Carga la red CIGRE desde Excel exportado de pandapower."""
    CIG_MV_NET_PATH = os.path.join(IN_PATH, 'CIG_MV_net_rev1_23_09_25.xlsx')
    net = pp.from_excel(CIG_MV_NET_PATH)
    return net

def decode_geodata_string(s):
    """Convierte un string JSON en dict con coordenadas."""
    if not isinstance(s, str):
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "coordinates" in obj:
            return obj["coordinates"]
    except Exception:
        pass
    return None

META = {
    'api_version': '3.0',
    'type': 'time-based',
    'models': {
        'PPModel': {
            'public': True,
            'params': [],
            'attrs': [
                'line_status', # PPModel output
                'fail_prob', # FailureModel input
                'repair_plan', # OpDecisionModel input
                'climate', # ClimateModel input
                'ens', # PPModel output (evaluation)
                'line_positions',
                'grid_x',
                'grid_y',
                'shape',
                'lines',
                'buses',
                'switches',
                'transformers',
                'loads',
                'switch_plan',
            ],
        }
    },
}


class PPModel(mosaik_api.Simulator):
    def __init__(self):
        super().__init__(META)
        self.eid = 'Grid'
        self.net = pp.create_empty_network()
        self.rng_sim = np.random.default_rng(2025)
        
        # Previous Config.
        self.line_status = {}
        
        # Inputs
        self.fail_prob = {}
        self.repair_plan = {}
        self.switch_plan = {}
        
        # Results
        self.logger = Logger("results/results.csv")
        self.lines = {}
        self.t = 0
        self.R_curve = {}
        self.switch_records = []
        self.ens_chosen = 0
        
        # MonteCarlo
        self.M = 30               # nº muestras MC por step (empieza con 20-30)
        self.alpha = 0.90         # nivel para VaR/CVaR
        self.mc_stats = []        # guardar P90/CVaR por hora
        
        # Plot Net
        self.enable_plot = False  # MUY importante en MC
        
        # -----------------------------
        # Inmunidad post-reparación
        # -----------------------------
        self.repair_immunity_s = 2 * 3600   # 2 horas (ajusta)
        self.immunity_until = {}            # {lid: time_s_hasta_el_que_es_inmune}

        
    # =============================================================
    # INITIALIZATION
    # =============================================================
        
    def init(self, sid, **sim_params):
        return META
    
    def create(self, num, model):
        """Crea entidad de red y configura topología."""
        
        print("\nRecibido network_data desde mosaik_config.py")
        self.setup_network()

        return [{'eid': self.eid, 'type': model, 'rel': []}]

    # ==============================================================
    # SETUP
    # ==============================================================

    def setup_network(self):
        """Carga la red CIGRE y convierte geodata escapado a coordenadas x,y."""
        
        print("Cargando red CIGRE desde Excel...")
        IN_PATH = "./data/red_electrica"

        # ============================
        # 1) Cargar la red
        # ============================
        try:
            self.net = net_cigre15_mv(IN_PATH)
        except Exception as e:
            print(f"Error cargando red CIGRE: {e}")
            raise e

        net = self.net
        
        # ============================================
        # SWITCHES SEGÚN dnr_status
        # ============================================
        print(f"Switches añadidos a líneas switcheables...")
        for lid in net.line.index:
            raw = str(net.line.at[lid, "dnr_status"]).lower()
            status = raw.replace("{", "").replace("}", "").strip()
            
            # ¿es switchable según el Excel?
            if status != "switchable":
                continue
            
            # Crear switch línea–bus en ambos extremos
            fb = net.line.at[lid, "from_bus"]
            tb = net.line.at[lid, "to_bus"]

            if lid not in net.switch.element.values:
                pp.create_switch(net, bus=fb, element=lid, et="l", closed=True)
                
        # ============================
        # 2) Decodificar geodata de buses
        # ============================
        if "geo" in net.bus.columns:
            print("Decodificando geodata de buses...")
            for b in net.bus.index:
                coords = decode_geodata_string(net.bus.at[b, "geo"])
                if coords:
                    net.bus.at[b, "x"] = float(coords[0])*500
                    net.bus.at[b, "y"] = float(coords[1])*500

        # ============================
        # 4) Preparar lista de líneas y estados
        # ============================
        self.lines = dict(zip(net.line.index, net.line.name))
        self.line_status = {lid: 1 for lid in self.lines.keys()}

        print("Setup_network() completado.\n")

    # ==============================================================
    # SIMULATION STEP
    # ==============================================================

    def step(self, time, inputs, max_advance):
        print("\n==============================")
        self.t = int(time / 3600)
        print(f"STEP t = {self.t} h")

        # Read inputs
        repair_plan = None
        gust_speed = 0

        if inputs:
            src = list(inputs.keys())[0]
            vals = inputs[src]
            
            if 'fail_prob' in vals:
                fail_prob_all = list(vals['fail_prob'].values())[0]  # dict {k: {lid: q}}

                # Usar SOLO el presente (k = 0)
                if isinstance(fail_prob_all, dict):
                    self.fail_prob = fail_prob_all.get(0, {})
                else:
                    self.fail_prob = {}
                
            if 'repair_plan' in vals:
                repair_plan = list(vals['repair_plan'].values())[0]
                self.repair_plan = repair_plan
                
            if 'switch_plan' in vals:
                switch_plan = list(vals['switch_plan'].values())[0]
                self.switch_plan = switch_plan
                
            for sid, state in self.switch_plan.items():
                self.net.switch.at[int(sid), "closed"] = bool(state)

            for lid, data in self.repair_plan.items():
                if data.get('finish_time',0) <= self.t * 3600:
                    self.line_status[lid] = 1
                    self.net.line.at[int(lid), "in_service"] = True
                    
                    # Inmunidad
                    self.immunity_until[lid] = time + self.repair_immunity_s
                    
            self.immunity_until = {lid: until for lid, until in self.immunity_until.items() if until > time}
                    

            if 'climate' in vals:
                gust_speed = list(vals["climate"].values())[0][0]["gust_speed"]
                self.last_gust_speed = np.array(gust_speed)
            if 'grid_x' in vals:
                self.grid_x = np.array(list(vals['grid_x'].values())[0])
            if 'grid_y' in vals:
                self.grid_y = np.array(list(vals['grid_y'].values())[0])
            if 'shape' in vals:
                self.shape = tuple(list(vals['shape'].values())[0])


        t = time / 3600.0
        
        # ==============================================================
        # MONTECARLO
        # ==============================================================
        
        # ===== Inicializar Parámetros =====
        expected_load = float(self.net.load.p_mw.sum())
        line_ids = list(self.net.line.index)  
        ens_samples = np.empty(self.M, dtype=float)
        cache_pf = {}  # key(bytes) -> ens_mw      
        
        # ===== Guardar estado base del step (antes de muestrear) =====
        base_in_service = self.net.line["in_service"].copy()
        base_status = self.line_status.copy()
        
        # ===== For de Montecarlo =====
        for m in range(self.M):
            
            # ===== Reset al estado base del step =====
            self.net.line["in_service"] = base_in_service
            self.line_status = base_status.copy()

            for _, lid in enumerate(line_ids):
                lid = int(lid)

                # ===== Si ya estaba caída, Permanece caída =====
                if base_status.get(lid, 1) == 0:
                    self.line_status[lid] = 0
                    continue
                
                # ===== Inmunidad =====
                if self.immunity_until.get(lid, 0) > time:
                    self.line_status[lid] = 1
                    continue
                
                # ===== Romper Linea de Forma Aleatoria =====
                p = float(self.fail_prob.get(lid, 0.0))
                self.line_status[lid] = 0 if (self.rng_sim.random() < p) else 1

            # ===== Clave compacta para Cache: bitstring -> bytes =====
            state_vec = np.array(
                [self.line_status.get(lid, 1) for lid in line_ids],
                dtype=np.uint8
            )
            key = np.packbits(state_vec).tobytes()

            # ===== Comprobación Misma Config. =====
            if key in cache_pf:
                ens_mw = cache_pf[key]
                ens_samples[m] = ens_mw
                served_load = expected_load - ens_mw
                
                for i, lid in enumerate(line_ids):
                    self.net.line.at[int(lid), "in_service"] = bool(self.line_status[lid])
                
                continue

            # ===== Aplicar Estados Sampleados =====
            for i, lid in enumerate(line_ids):
                self.net.line.at[int(lid), "in_service"] = bool(self.line_status[lid])

            # ===== PF =====
            try:
                pp.runpp(self.net)
                served_load = float(self.net.res_load.p_mw.sum())
                ens_mw = max(0.0, expected_load - served_load)
            except Exception:
                ens_mw = expected_load

            # guardar en cache y en muestras
            cache_pf[key] = ens_mw
            ens_samples[m] = ens_mw
            
        # Actualizar líneas según el modelo de fallo
        for lid, status in self.line_status.items():
            self.net.line.at[lid, 'in_service'] = bool(status)
            
        for sw, state in self.switch_plan.items(): 
                line_id = self.net.switch.at[sw, "element"] 
                sw_in_service = bool(self.net.line.at[line_id, "in_service"]) 
                self.switch_records.append({ "time": t, "switch": sw, "state": state, "sw_in_service": sw_in_service })

        # VaR (P90) y CVaR0.9
        p90 = float(np.quantile(ens_samples, self.alpha))
        tail = ens_samples[ens_samples >= p90]
        cvar90 = float(tail.mean()) if tail.size else p90
        
        # R-Curve CVaR90
        R = (expected_load-cvar90)/expected_load
        # R = served_load/expected_load
        self.R_curve[self.t] = R
        self.ens_chosen = max(0.0, expected_load - served_load)   

        # Guardar resultados y plot
        if self.enable_plot:
            plot_network(self.net,
                         self.last_gust_speed, 
                         self.grid_x, 
                         self.grid_y, 
                         self.line_status, 
                         self.t)
        
        df_switches = pd.DataFrame(self.switch_records)
        df_switches.to_csv("results/switch_states_GA_PRUEBA.csv", index=False)
        
        if time >= (24 * 3600) - 3600:
            # último paso de simulación
            plot_R_curve(self.R_curve)
        
        self.mc_stats.append({
            "t": self.t,
            "ens_mean": float(ens_samples.mean()),
            "ens_p50": float(np.quantile(ens_samples, 0.50)),
            "ens_p90": p90,
            "ens_cvar90": cvar90,
            "ens_max": float(ens_samples.max()),
            "ens_chosen": float(self.ens_chosen),
        })
        
        print(f"[MC-DC] ENS mean={ens_samples.mean():.3f} MW | P90={p90:.3f} MW | CVaR0.9={cvar90:.3f} MW | chosen={self.ens_chosen:.3f} MW")

        return time + 3600

    # ==============================================================
    # UTILIDADES
    # ==============================================================

    def save_results(self, hour, wind_speed, ens):
        filename = f"results/hour_{hour:02d}.csv"
        fieldnames = ["hour", "wind_speed", "ens"] + [str(l) for l in self.lines.keys()]

        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            row = {"hour": hour, "wind_speed": wind_speed, "ens": ens}
            row.update({str(l): self.line_status.get(l, 1) for l in self.lines.keys()})
            writer.writerow(row)

    def get_data(self, outputs=None):
        """Devuelve ENS, corrientes y posiciones de línea a mosaik."""
        line_pos = {}
        for i, row in self.net.line.iterrows():
            bus0 = self.net.bus.at[row.from_bus, 'name']
            bus1 = self.net.bus.at[row.to_bus, 'name']
            
            # Leer geodata de forma segura. Pandapower guarda la tupla en la columna 'geodata'.
            def _get_geodata(idx):
                """
                Transforms data from pp dictionary and makes it plottable
                From str to float

                Args:
                    idx: id of the bus

                Returns:
                    tuple containing position of the bus in float
                """
                if 'x' in self.net.bus.columns and 'y' in self.net.bus.columns:
                    try:
                        return (float(self.net.bus.at[idx, 'x']), float(self.net.bus.at[idx, 'y']))
                    except Exception:
                        return (0.0, 0.0)
                return (0.0, 0.0)

            x0, y0 = _get_geodata(row.from_bus)
            x1, y1 = _get_geodata(row.to_bus)
            line_pos[row.name] = {"bus0": bus0, "bus1": bus1, "x0": x0, "y0": y0, "x1": x1, "y1": y1}

        return {
            self.eid: {
                'ens': self.ens_chosen,
                'line_positions': line_pos,
                'line_status': self.line_status,
                'lines': self.net.line,
                'buses': self.net.bus,
                'switches': self.net.switch,
                'transformers': self.net.trafo,
                'loads': self.net.load,
            }
        }