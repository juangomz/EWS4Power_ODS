import gurobipy as gp
from gurobipy import GRB

def optimize_switches_gurobi(self_buses, self_lines, self_switches_buses, self_transformers, self_fail_prob, repair_lines=None,):
        if repair_lines is None: repair_lines = []

        model = gp.Model("switch_radial_final_fix")
        model.Params.OutputFlag = 0

        # ========= CONJUNTOS =========
        buses = list(self_buses.index)
        lines = list(self_lines.index)
        switches = list(self_switches_buses.index)
        transformers = list(self_transformers.index)
        
        BIG_M = len(buses) + 5

        # ========= VARIABLES =========
        e = {b: model.addVar(vtype=GRB.BINARY, name=f"e_{b}") for b in buses}
        x = {sid: model.addVar(vtype=GRB.BINARY, name=f"x_sw_{sid}") for sid in switches}
        
        y_line = {lid: model.addVar(vtype=GRB.BINARY) for lid in lines}
        y_sw = {sid: model.addVar(vtype=GRB.BINARY) for sid in switches}
        y_tr = {tid: model.addVar(vtype=GRB.BINARY) for tid in transformers}

        beta_line = {(lid, d): model.addVar(vtype=GRB.BINARY) for lid in lines for d in [0, 1]}
        beta_sw = {(sid, d): model.addVar(vtype=GRB.BINARY) for sid in switches for d in [0, 1]}
        beta_tr = {(tid, d): model.addVar(vtype=GRB.BINARY) for tid in transformers for d in [0, 1]}

        u = {b: model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=len(buses)) for b in buses}

        # ========= SLACK =========
        try:
            slack = self_buses[self_buses["name"] == "Bus 0"].index[0]
        except:
            slack = buses[0]
        model.addConstr(e[slack] == 1)
        model.addConstr(u[slack] == 0)

        # ========= RESTRICCIONES =========
        
        # --- LÍNEAS (Lógica de Cable Sólido Condicional) ---
        for lid in lines:
            # 1. Definición Básica
            model.addConstr(y_line[lid] == beta_line[(lid, 0)] + beta_line[(lid, 1)])
            
            fb, tb = int(self_lines.at[lid, "from_bus"]), int(self_lines.at[lid, "to_bus"])
            in_service = bool(self_lines.at[lid, "in_service"])
            is_broken = (lid in repair_lines) or (not in_service)
            
            # 2. Lógica Física
            if is_broken:
                # CASO A: Rota -> Cortada obligatoriamente
                model.addConstr(y_line[lid] == 0, name=f"broken_{lid}")
            else:
                # CASO B: Sana -> Cable Sólido
                # Si fb=1 y tb=1, entonces y=1. 
                # Esto PROHÍBE al optimizador cortar esta línea para romper bucles.
                model.addConstr(y_line[lid] >= e[fb] + e[tb] - 1, name=f"solid_{lid}")

            # 3. Consistencia Dirección
            model.addConstr(beta_line[(lid, 0)] <= e[fb])
            model.addConstr(beta_line[(lid, 1)] <= e[tb])

            # 4. Profundidad (MTZ)
            model.addConstr(u[tb] >= u[fb] + 1 - BIG_M * (1 - beta_line[(lid, 0)]))
            model.addConstr(u[fb] >= u[tb] + 1 - BIG_M * (1 - beta_line[(lid, 1)]))

        # --- SWITCHES ---
        for sid in switches:
            model.addConstr(y_sw[sid] == beta_sw[(sid, 0)] + beta_sw[(sid, 1)])
            
            # Igualdad Estricta (x=y)
            # NOTA: Aquí NO ponemos 'solid_wire'. Los switches SÍ pueden abrirse 
            # aunque ambos lados tengan luz (Punto de Apertura Normal).
            model.addConstr(x[sid] == y_sw[sid])

            a, b = int(self_switches_buses.at[sid, "bus"]), int(self_switches_buses.at[sid, "element"])
            model.addConstr(beta_sw[(sid, 0)] <= e[a])
            model.addConstr(beta_sw[(sid, 1)] <= e[b])

            # Profundidad (MTZ)
            model.addConstr(u[b] >= u[a] + 1 - BIG_M * (1 - beta_sw[(sid, 0)]))
            model.addConstr(u[a] >= u[b] + 1 - BIG_M * (1 - beta_sw[(sid, 1)]))

        # --- TRANSFORMADORES ---
        for tid in transformers:
            model.addConstr(y_tr[tid] == beta_tr[(tid, 0)] + beta_tr[(tid, 1)])
            fb, tb = int(self_transformers.at[tid, "hv_bus"]), int(self_transformers.at[tid, "lv_bus"])
            model.addConstr(beta_tr[(tid, 0)] <= e[fb])
            model.addConstr(beta_tr[(tid, 1)] <= e[tb])
            
            # Profundidad (MTZ)
            model.addConstr(u[tb] >= u[fb] + 1 - BIG_M * (1 - beta_tr[(tid, 0)]))
            model.addConstr(u[fb] >= u[tb] + 1 - BIG_M * (1 - beta_tr[(tid, 1)]))

        # --- PADRE ÚNICO ---
        for b in buses:
            incoming = [] 
            for lid in lines:
                fb, tb = int(self_lines.at[lid, "from_bus"]), int(self_lines.at[lid, "to_bus"])
                if tb == b: incoming.append(beta_line[(lid, 0)]) 
                if fb == b: incoming.append(beta_line[(lid, 1)]) 
            for sid in switches:
                fb, tb = int(self_switches_buses.at[sid, "bus"]), int(self_switches_buses.at[sid, "element"])
                if tb == b: incoming.append(beta_sw[(sid, 0)])
                if fb == b: incoming.append(beta_sw[(sid, 1)])
            for tid in transformers:
                fb, tb = int(self_transformers.at[tid, "hv_bus"]), int(self_transformers.at[tid, "lv_bus"])
                if tb == b: incoming.append(beta_tr[(tid, 0)])
                if fb == b: incoming.append(beta_tr[(tid, 1)])

            if b == slack:
                model.addConstr(gp.quicksum(incoming) == 0)
            else:
                model.addConstr(gp.quicksum(incoming) == e[b])

        # ========= OBJETIVO =========
        risk_term = gp.quicksum(
            float(self_fail_prob.get(lid, 0.0)) * y_line[lid] 
            for lid in lines
        )
        
        BETA = 0.001
        ALPHA = 1 
        
        obj = gp.quicksum(e[b] for b in buses) \
              - ALPHA * risk_term \
              - BETA * gp.quicksum(x[sid] for sid in switches)
        
        model.setObjective(obj, GRB.MAXIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            print("\n" + "="*60)
            print("🕵️‍♂️ REPORTE DE DECISIÓN DEL OPTIMIZADOR")
            print("="*60)
            
            # 1. ANÁLISIS DE OBJETIVO
            total_buses = len(buses)
            energized_count = sum(round(e[b].X) for b in buses)
            total_risk = risk_term.getValue()
            total_switches_closed = sum(round(x[sid].X) for sid in switches)
            
            print(f"\n📊 RESUMEN GENERAL:")
            print(f"   - Buses Energizados: {energized_count}/{total_buses} (+{energized_count} pts)")
            print(f"   - Riesgo Acumulado:  {total_risk:.4f} (-{ALPHA * total_risk:.4f} pts)")
            print(f"   - Switches Cerrados: {total_switches_closed} (-{BETA * total_switches_closed:.4f} pts)")
            print(f"   - FUNCIÓN OBJETIVO:  {model.ObjVal:.4f}")

            # 2. ANÁLISIS DE RUTAS (TRACE-BACK)
            # Vamos a rastrear de dónde le llega la luz a cada bus clave
            print(f"\n📍 RASTREO DE CONEXIONES (¿Quién alimenta a quién?):")
            
            # Construimos un mapa de padres para imprimir
            parent_map = {}
            for lid in lines:
                if y_line[lid].X > 0.5:
                    fb, tb = int(self_lines.at[lid, "from_bus"]), int(self_lines.at[lid, "to_bus"])
                    # Si beta_0=1, fb es padre de tb
                    if beta_line[(lid, 0)].X > 0.5:
                        parent_map[tb] = (fb, "Line", lid, self_fail_prob.get(lid, 0.0))
                    else:
                        parent_map[fb] = (tb, "Line", lid, self_fail_prob.get(lid, 0.0))
            
            for sid in switches:
                if y_sw[sid].X > 0.5:
                    fb, tb = int(self_switches_buses.at[sid, "bus"]), int(self_switches_buses.at[sid, "element"])
                    if beta_sw[(sid, 0)].X > 0.5:
                        parent_map[tb] = (fb, "Switch", sid, 0.0) # Switches no tienen riesgo en tu modelo actual
                    else:
                        parent_map[fb] = (tb, "Switch", sid, 0.0)

            for tid in transformers:
                 if y_tr[tid].X > 0.5:
                    fb, tb = int(self_transformers.at[tid, "hv_bus"]), int(self_transformers.at[tid, "lv_bus"])
                    if beta_tr[(tid, 0)].X > 0.5:
                        parent_map[tb] = (fb, "Trafo", tid, 0.0)
                    else:
                        parent_map[fb] = (tb, "Trafo", tid, 0.0)

            # Imprimir jerarquía para buses energizados
            for b in sorted(buses):
                if round(e[b].X) == 1:
                    if b == slack:
                        print(f"   Bus {b}: 👑 SLACK (Fuente)")
                    elif b in parent_map:
                        padre, tipo, id_elem, prob = parent_map[b]
                        riesgo_txt = f"(Riesgo: {prob:.2f})" if tipo == "Line" else "(Riesgo: 0)"
                        print(f"   Bus {b} <--- {tipo} {id_elem} {riesgo_txt} --- Bus {padre}")
                    else:
                        print(f"   Bus {b}: ⚠️ Energizado pero sin padre (¿Error?)")
                else:
                    print(f"   Bus {b}: 🌑 APAGADO")

            # 3. ANÁLISIS DE CORTES (¿Por qué abrió aquí?)
            print(f"\n✂️ PUNTOS DE CORTE (Donde se rompen los bucles):")
            
            # Líneas sanas pero abiertas (y=0, e_a=1, e_b=1)
            for lid in lines:
                fb, tb = int(self_lines.at[lid, "from_bus"]), int(self_lines.at[lid, "to_bus"])
                is_broken = (lid in repair_lines) or (not bool(self_lines.at[lid, "in_service"]))
                
                if not is_broken and y_line[lid].X < 0.5:
                    print(f"   ❌ Line {lid} ({fb}-{tb}): ABIERTA por el optimizador.")
                    print(f"      Estado: Ambos extremos tienen luz? {round(e[fb].X)} - {round(e[tb].X)}")
                    if round(e[fb].X) == 1 and round(e[tb].X) == 1:
                         print("      👉 ¡AQUI ESTÁ EL CORTE DEL BUCLE! (Open Point)")
                    else:
                         print("      👉 Abierta porque una zona está apagada.")
            
            # Líneas rotas
            for lid in lines:
                is_broken = (lid in repair_lines) or (not bool(self_lines.at[lid, "in_service"]))
                if is_broken:
                     print(f"   🛠️ Line {lid}: ROTAS/REPARACIÓN (Forzada a 0)")

            # Switches abiertos
            for sid in switches:
                if x[sid].X < 0.5:
                    a, b = int(self_switches_buses.at[sid, "bus"]), int(self_switches_buses.at[sid, "element"])
                    print(f"   🔓 Switch {sid} ({a}-{b}): ABIERTO")
                    if round(e[a].X) == 1 and round(e[b].X) == 1:
                        print("      👉 ¡PUNTO DE APERTURA DE ANILLO!")

            print("="*60 + "\n")
            
            return {sid: int(round(x[sid].X)) for sid in switches}
        else:
            print("Optimización falló.")
            return {}
    