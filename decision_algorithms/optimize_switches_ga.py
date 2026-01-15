import networkx as nx
import random

def optimize_switches_ga(
        self_buses,
        self_lines,
        self_switches_buses,
        self_transformers,
        self_loads,
        self_fail_prob,
        prev_switch_state=None,
        population_size=60,
        generations=60,
        crossover_rate=0.8,
        mutation_rate=0.01,   # 🔹 mutación más suave (warm-start)
        lambda_sw_op = 0,
        lambda_sw_removed=0,
        lambda_bus_disconnected=0,
        plot=True,
        seed=1234
    ):
    """
    Algoritmo genético para optimizar los estados de los switches.
    Warm-start desde la topología previa almacenada en self_switches_buses.
    """
    
    rng = random.Random(seed)

    # =============================
    # CONJUNTOS
    # =============================
    buses = list(self_buses.index)
    switches = list(self_switches_buses.index)
    lines = list(self_lines.index)
    transformers = list(self_transformers.index)
    loads = list(self_loads.index)
    
    # Precalcular la potencia por bus
    p_load_per_bus = (
    self_loads
    .groupby("bus")["p_mw"]
    .sum()
    .to_dict()
)

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
        
    if self_fail_prob is None:
        # Modo reactivo: un único "paso" sin riesgo
        fail_prob_iter = {0: {}}
    else:
        fail_prob_iter = self_fail_prob
        
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

    # Transformadores
    for tid in transformers:
        fb = int(self_transformers.at[tid, "hv_bus"])
        tb = int(self_transformers.at[tid, "lv_bus"])
        G.add_edge(fb, tb, tipo="trafo", tid=tid)

    # =============================
    # FITNESS
    # =============================
    def evaluate_fitness(switch_state, prev_switch_state):
        total_score = 0.0
        total_penalty_b_sin_conectar = 0.0
        total_penalty_s_removed = 0.0
        
        # --- Evaluar sobre todo el horizonte a estudiar ---
        for k, fail_prob_k in fail_prob_iter.items():
            
            p_line = {
                lid: max(1.0 - fail_prob_k.get(lid, 0.0), 1e-6)
                for lid in lines
            }

            # Switches
            effective_ind = switch_state.copy()
            for i, sid in enumerate(switches):
                if switch_state[i] != 1:
                    continue

                a = int(self_switches_buses.at[sid, "bus"])
                b = int(self_switches_buses.at[sid, "element"])

                lid = line_map.get((a, b))
                if lid is None:
                    effective_ind[i] = 0
                    continue

                if not bool(self_lines.at[lid, "in_service"]):
                    effective_ind[i] = 0
                    continue

                # SOLO ahora la arista existe
                G.add_edge(a, b, tipo="line", lid=lid, sid=sid)
                
            sid_index = {sid: i for i, sid in enumerate(switches)}

            # Componente slack
            if slack not in G:
                return -1e3

            comp = nx.node_connected_component(G, slack)
            H = G.subgraph(comp).copy()

            # Reparar topología: romper ciclos antes de evaluar        
            H, removed_sids = repair_to_radial(H, p_line)
            
            # crear cromosoma reparado
            radial_ind = effective_ind.copy()
            for sid in removed_sids:
                radial_ind[sid_index[sid]] = 0

            # =============================
            # Penalizaciones (CLAVE)
            # =============================
            penalty_b_sin_conectar = 0.0
            penalty_s_removed = 0.0

            # nodos no conectados
            penalty_b_sin_conectar += (len(buses) - len(H.nodes()))
            penalty_s_removed += len(removed_sids)

            # =============================
            # Score por terminales
            # =============================
            score = 0.0

            for b in buses:

                if b == slack or b not in H.nodes():
                    continue

                path = nx.shortest_path(H, slack, b)

                p_bus = p_load_per_bus.get(b, 0.0)
                
                for u, v in zip(path[:-1], path[1:]):
                    e = H.get_edge_data(u, v)
                    if e["tipo"] == "trafo":
                        p_bus *= 1
                    else:
                        p_bus *= p_line[e["lid"]]

                score += p_bus
                
            total_score += score
            total_penalty_b_sin_conectar += penalty_b_sin_conectar
            total_penalty_s_removed += penalty_s_removed
        
        # coste de switching
        switch_cost = sum(
            abs(radial_ind[i] - prev_switch_state[i]) for i in range(len(radial_ind))
        )

        return total_score - lambda_bus_disconnected*total_penalty_b_sin_conectar - lambda_sw_removed*total_penalty_s_removed - lambda_sw_op*switch_cost, radial_ind, switch_cost
    
    # =============================
    # DISTANCIA DE HAMMING
    # =============================
    def switching_cost(x, x_prev):
        return sum(abs(a - b) for a, b in zip(x, x_prev))

    
    # =============================
    # ASEGUTRADOR DE RADIALIDAD
    # =============================
    def repair_to_radial(H, p_line):
        

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
        if rng.random() < 0.3:
            ones = [i for i,v in enumerate(ind) if v==1]
            zeros = [i for i,v in enumerate(ind) if v==0]
            if ones and zeros:
                i_off = rng.choice(ones)
                i_on  = rng.choice(zeros)
                ind[i_off] = 0
                ind[i_on]  = 1

        # y además un flip suave
        for i in range(len(ind)):
            if rng.random() < mutation_rate:
                ind[i] = 1 - ind[i]
        return ind

    def crossover(p1, p2):
        if rng.random() > crossover_rate:
            return p1.copy(), p2.copy()
        point = rng.randint(1, len(p1) - 1)
        return (
            p1[:point] + p2[point:],
            p2[:point] + p1[point:]
        )

    def tournament_selection(pop, fits, k=3):
        best = None
        best_fit = -1e12
        for _ in range(k):
            i = rng.randint(0, len(pop) - 1)
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
        return [rng.randint(0, 1) for _ in switches]

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
            if rng.random() < 0.2:
                new_pop.append(random_individual())
                continue
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            h1, h2 = crossover(p1, p2)
            new_pop.append(mutate(h1))
            if len(new_pop) < population_size:
                new_pop.append(mutate(h2))

        population = new_pop
    
    if plot:
        import os
        import numpy as np
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        os.makedirs("figures/GA", exist_ok=True)

        mpl.rcParams.update({
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

        hist = np.array(history, dtype=float)

        # Best-so-far (monótono) para lectura de convergencia
        best_so_far = np.maximum.accumulate(hist)

        fig, ax = plt.subplots(figsize=(6.2, 2.8))
        ax.plot(hist, linewidth=1.0, alpha=0.5, label="Best of generation")
        ax.plot(best_so_far, linewidth=1.6, label="Best-so-far")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("GA convergence (warm-start)")
        ax.grid(alpha=0.25)

        ax.legend(frameon=True, framealpha=0.95, fontsize=8, loc="lower right")
        fig.tight_layout()

        fig.savefig("figures/GA/convergence.pdf", bbox_inches="tight")
        fig.savefig("figures/GA/convergence.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


    return {
        "switch_state": {sid: best_global[i] for i, sid in enumerate(switches)},
        "fitness": best_global_fit,
        "switches_changed": switches_changed
    }
    