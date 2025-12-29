def optimize_switches_ga(
        self_buses,
        self_lines,
        self_switches_buses,
        self_transformers,
        self_fail_prob,
        prev_switch_state=None,
        population_size=60,
        generations=60,
        crossover_rate=0.8,
        mutation_rate=0.01,   # 🔹 mutación más suave (warm-start)
        lambda_sw = 0,
        plot=True
    ):
    """
    Algoritmo genético para optimizar los estados de los switches.
    Warm-start desde la topología previa almacenada en self_switches_buses.
    """

    import random
    import networkx as nx
    import matplotlib.pyplot as plt

    # =============================
    # CONJUNTOS
    # =============================
    buses = list(self_buses.index)
    switches = list(self_switches_buses.index)
    lines = list(self_lines.index)
    transformers = list(self_transformers.index)

    # Identificar slack
    try:
        slack = self_buses[self_buses["name"] == "Bus 0"].index[0]
    except:
        slack = buses[0]

    # =============================
    # DETECTAR TERMINALES
    # =============================
    degree = {b: 0 for b in buses}

    for lid in lines:
        fb = int(self_lines.at[lid, "from_bus"])
        tb = int(self_lines.at[lid, "to_bus"])
        degree[fb] += 1
        degree[tb] += 1

    for sid in switches:
        fb = int(self_switches_buses.at[sid, "bus"])
        tb = int(self_switches_buses.at[sid, "element"])
        degree[fb] += 1
        degree[tb] += 1

    for tid in transformers:
        fb = int(self_transformers.at[tid, "hv_bus"])
        tb = int(self_transformers.at[tid, "lv_bus"])
        degree[fb] += 1
        degree[tb] += 1

    # =============================
    # FITNESS
    # =============================
    def evaluate_fitness(switch_state, prev_switch_state):
        total_score = 0.0
        total_penalty = 0.0
        for k, fail_prob_k in self_fail_prob.items():
            
            p_line = {
                lid: max(1.0 - fail_prob_k.get(lid, 0.0), 1e-6)
                for lid in lines
            }
            
            G = nx.Graph()
            for b in buses:
                G.add_node(b)

            # Líneas
            line_map = {}
            for lid in lines:
                fb = int(self_lines.at[lid, "from_bus"])
                tb = int(self_lines.at[lid, "to_bus"])
                line_map[(fb, tb)] = lid
                line_map[(tb, fb)] = lid  # importante

            # Switches
            for i, sid in enumerate(switches):
                if switch_state[i] != 1:
                    continue  # switch abierto → NO hay arista

                a = int(self_switches_buses.at[sid, "bus"])
                b = int(self_switches_buses.at[sid, "element"])

                lid = line_map.get((a, b))
                if lid is None:
                    continue

                # comprobar si la línea física está viva
                if not bool(self_lines.at[lid, "in_service"]):
                    continue

                # SOLO ahora la arista existe
                G.add_edge(a, b, tipo="line", lid=lid, sid=sid)
            sid_index = {sid: i for i, sid in enumerate(switches)}

            # Transformadores
            for tid in transformers:
                fb = int(self_transformers.at[tid, "hv_bus"])
                tb = int(self_transformers.at[tid, "lv_bus"])
                G.add_edge(fb, tb, tipo="trafo", tid=tid)

            # Componente slack
            if slack not in G:
                return -1e3

            comp = nx.node_connected_component(G, slack)
            H = G.subgraph(comp).copy()

            # 🔧 Reparar topología: romper ciclos antes de evaluar        
            H, removed_sids = repair_to_radial(H, p_line)
            
            # 🔧 crear cromosoma reparado
            repaired_ind = switch_state.copy()
            for sid in removed_sids:
                repaired_ind[sid_index[sid]] = 0

            # =============================
            # Penalizaciones (CLAVE)
            # =============================
            penalty = 0.0

            # nodos no conectados
            penalty += (len(buses) - len(H.nodes())) + len(removed_sids)

            # =============================
            # Score por terminales
            # =============================
            score = 0.0

            for b in buses:

                if b == slack or b not in H.nodes():
                    continue

                path = nx.shortest_path(H, slack, b)

                p_bus = 1.0
                for u, v in zip(path[:-1], path[1:]):
                    e = H.get_edge_data(u, v)
                    if e["tipo"] == "trafo":
                        p_bus *= 1
                    else:
                        p_bus *= p_line[e["lid"]]

                score += p_bus
                
            total_score += score
            total_penalty += penalty
        
        # coste de switching
        switch_cost = sum(
            abs(repaired_ind[i] - prev_switch_state[i]) for i in range(len(repaired_ind))
        )

        return total_score - total_penalty - lambda_sw*switch_cost, repaired_ind, switch_cost
    
    # =============================
    # DISTANCIA DE HAMMING
    # =============================
    def switching_cost(x, x_prev):
        return sum(abs(a - b) for a, b in zip(x, x_prev))

    
    # =============================
    # ASEGUTRADOR DE RADIALIDAD
    # =============================
    def repair_to_radial(H, p_line):
        import networkx as nx

        H = H.copy()
        removed_sids = set()

        while True:
            try:
                cycle = nx.find_cycle(H)
            except nx.exception.NetworkXNoCycle:
                break

            worst_edge = None
            worst_q = -1.0

            for u, v in cycle:
                attrs = H.get_edge_data(u, v)

                if attrs["tipo"] == "line":
                    lid = attrs["lid"]
                    q = 1.0 - p_line.get(lid, 1.0)
                else:
                    q = 0.0

                if q > worst_q:
                    worst_q = q
                    worst_edge = (u, v)

            u, v = worst_edge
            attrs = H.get_edge_data(u, v)
            if "sid" in attrs:
                removed_sids.add(attrs["sid"])

            H.remove_edge(u, v)

        return H, removed_sids


    # =============================
    # OPERADORES GENÉTICOS
    # =============================

    def mutate(ind):
        # mutación swap con cierta probabilidad
        if random.random() < 0.3:
            ones = [i for i,v in enumerate(ind) if v==1]
            zeros = [i for i,v in enumerate(ind) if v==0]
            if ones and zeros:
                i_off = random.choice(ones)
                i_on  = random.choice(zeros)
                ind[i_off] = 0
                ind[i_on]  = 1

        # y además un flip suave
        for i in range(len(ind)):
            if random.random() < mutation_rate:
                ind[i] = 1 - ind[i]
        return ind

    def crossover(p1, p2):
        if random.random() > crossover_rate:
            return p1.copy(), p2.copy()
        point = random.randint(1, len(p1) - 1)
        return (
            p1[:point] + p2[point:],
            p2[:point] + p1[point:]
        )

    def tournament_selection(pop, fits, k=3):
        best = None
        best_fit = -1e12
        for _ in range(k):
            i = random.randint(0, len(pop) - 1)
            if fits[i] > best_fit:
                best_fit = fits[i]
                best = pop[i]
        return best.copy()

    # =============================
    # 🔹 WARM-START DESDE LA RED
    # =============================
    def base_individual_from_network():
        ind = []
        for sid in switches:
            ind.append(int(self_switches_buses.at[sid, "closed"]))
        return ind

    base_ind = base_individual_from_network()
    
    if prev_switch_state is None:
        prev_switch_state = base_ind.copy()
    
    immigrant_rate = 0.3
    population = [base_ind.copy()]
    n_imm = int((population_size - 1) * immigrant_rate)

    # vecinos mutados
    while len(population) < population_size - n_imm:
        ind = base_ind.copy()
        population.append(mutate(ind))

    # inmigrantes aleatorios
    def random_individual():
        return [random.randint(0, 1) for _ in switches]

    while len(population) < population_size:
        population.append(random_individual())

    # =============================
    # LOOP GA
    # =============================
    best_global = base_ind.copy()
    best_global_fit, best_global, switches_changed = evaluate_fitness(best_global, prev_switch_state)
    history = []
    
    for gen in range(generations):

        evaluated = [evaluate_fitness(ind, prev_switch_state=prev_switch_state) for ind in population]
        fitness = [e[0] for e in evaluated]
        repaired_inds = [e[1] for e in evaluated]
        switch_costs = [e[2] for e in evaluated]

        idx = fitness.index(max(fitness))
        history.append(fitness[idx])

        if fitness[idx] > best_global_fit:
            best_global_fit = fitness[idx]
            best_global = repaired_inds[idx].copy()
            switches_changed = switch_costs[idx]

        new_pop = [best_global.copy()]
            
        while len(new_pop) < population_size:
            if random.random() < 0.2:
                new_pop.append(random_individual())
                continue
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            h1, h2 = crossover(p1, p2)
            new_pop.append(mutate(h1))
            if len(new_pop) < population_size:
                new_pop.append(mutate(h2))

        population = new_pop

    # =============================
    # PLOT
    # =============================
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(history)
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.title("Convergencia GA (warm-start)")
        plt.grid(True)
        plt.savefig("figures/GA/convergence.png")
        plt.close()

    return {
        "switch_state": {sid: best_global[i] for i, sid in enumerate(switches)},
        "fitness": best_global_fit,
        "switches_changed": switches_changed
    }
