def optimize_switches_ga(
        self_buses,
        self_lines,
        self_switches_buses,
        self_transformers,
        self_fail_prob,
        population_size=60,
        generations=120,
        crossover_rate=0.8,
        mutation_rate=0.05,
        verbose=False,
        plot=True
    ):
    """
    Algoritmo genético para optimizar los estados de los switches.
    Criterio: maximizar la probabilidad total de suministro en nodos terminales.

    - Los transformadores NO son reconfigurables, pero afectan a la topología.
    - La radialidad SOLO se exige en la componente conectada al slack.
    - Los buses no alcanzables simplemente no aportan probabilidad.
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
    # PRE-CALCULAR p_l = 1 - q_l
    # =============================
    p_line = {}
    for lid in lines:
        q = float(self_fail_prob.get(lid, 0.0))
        p_line[lid] = max(1.0 - q, 1e-6)

    # =============================
    # DETECTAR TERMINALES
    # (en base a topología estática)
    # =============================
    degree = {b: 0 for b in buses}

    # Líneas
    for lid in lines:
        fb = int(self_lines.at[lid, "from_bus"])
        tb = int(self_lines.at[lid, "to_bus"])
        degree[fb] += 1
        degree[tb] += 1

    # Switches
    for sid in switches:
        fb = int(self_switches_buses.at[sid, "bus"])
        tb = int(self_switches_buses.at[sid, "element"])
        degree[fb] += 1
        degree[tb] += 1

    # Transformadores
    for tid in transformers:
        fb = int(self_transformers.at[tid, "hv_bus"])
        tb = int(self_transformers.at[tid, "lv_bus"])
        degree[fb] += 1
        degree[tb] += 1

    terminal_buses = [b for b in buses if b != slack and degree[b] == 1]

    # =============================
    # FUNCIÓN DE FITNESS
    # =============================
    def evaluate_fitness(switch_state):
        """
        switch_state: lista binaria con longitud len(switches)
        """

        # 1. Construir grafo
        G = nx.Graph()
        for b in buses:
            G.add_node(b)

        # Líneas en servicio
        for lid in lines:
            fb = int(self_lines.at[lid, "from_bus"])
            tb = int(self_lines.at[lid, "to_bus"])
            if bool(self_lines.at[lid, "in_service"]):
                G.add_edge(fb, tb, tipo="line", lid=lid)

        # Switches según cromosoma
        for i, sid in enumerate(switches):
            if switch_state[i] == 1:
                a = int(self_switches_buses.at[sid, "bus"])
                b = int(self_switches_buses.at[sid, "element"])
                G.add_edge(a, b, tipo="sw", sid=sid)

        # Transformadores (fijos)
        for tid in transformers:
            fb = int(self_transformers.at[tid, "hv_bus"])
            tb = int(self_transformers.at[tid, "lv_bus"])
            G.add_edge(fb, tb, tipo="trafo", tid=tid)

        # 2. Componente conectada al slack
        comp = nx.node_connected_component(G, slack)
        H = G.subgraph(comp).copy()

        # 3. Radialidad SOLO en H
        # Un árbol cumple |E| = |V| - 1
        if H.number_of_edges() != H.number_of_nodes() - 1:
            return -1.0  # no radial → mala solución

        # 4. Probabilidad acumulada en terminales alcanzados
        score = 0.0

        for b in terminal_buses:
            if b not in H.nodes():
                continue

            path = nx.shortest_path(H, slack, b)

            p_bus = 1.0
            for u, v in zip(path[:-1], path[1:]):
                e = H.get_edge_data(u, v)
                if e["tipo"] == "line":
                    lid = e["lid"]
                    p_bus *= p_line[lid]

            score += p_bus

        return score

    # =============================
    # FUNCIONES GENÉTICAS
    # =============================
    def random_individual():
        return [random.randint(0, 1) for _ in switches]

    def crossover(parent1, parent2):
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, len(switches) - 1)
        h1 = parent1[:point] + parent2[point:]
        h2 = parent2[:point] + parent1[point:]
        return h1, h2

    def mutate(ind):
        for i in range(len(ind)):
            if random.random() < mutation_rate:
                ind[i] = 1 - ind[i]
        return ind

    def tournament_selection(pop, fits, k=3):
        best = None
        best_fit = -1e9
        for _ in range(k):
            idx = random.randint(0, len(pop)-1)
            if fits[idx] > best_fit:
                best_fit = fits[idx]
                best = pop[idx]
        return best.copy()

    # =============================
    # INICIALIZAR POBLACIÓN
    # =============================
    population = [random_individual() for _ in range(population_size)]

    best_global = None
    best_global_fit = -1.0
    history = []

    # =============================
    # LOOP PRINCIPAL DEL GA
    # =============================
    for gen in range(generations):

        fitness = [evaluate_fitness(ind) for ind in population]

        gen_best_fit = max(fitness)
        gen_best_ind = population[fitness.index(gen_best_fit)]

        history.append(gen_best_fit)

        if gen_best_fit > best_global_fit:
            best_global_fit = gen_best_fit
            best_global = gen_best_ind.copy()

        if verbose:
            print(f"Generación {gen}: mejor fitness = {gen_best_fit:.6f}")

        # Nueva población con elitismo
        new_pop = [best_global.copy()]

        while len(new_pop) < population_size:
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            h1, h2 = crossover(p1, p2)
            new_pop.append(mutate(h1))
            if len(new_pop) < population_size:
                new_pop.append(mutate(h2))
        population = new_pop

    # =============================
    # PLOTEAR CONVERGENCIA
    # =============================
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(history, label="Mejor fitness por generación")
        plt.xlabel("Generación")
        plt.ylabel("Fitness (probabilidad total terminales)")
        plt.title("Evolución del Algoritmo Genético")
        plt.grid(True)
        plt.legend()
        plt.savefig("figures/GA/convergence.png")

    # =============================
    # RESULTADO FINAL
    # =============================
    if verbose:
        print("\n🔍 Mejor solución encontrada:")
        print(best_global)
        print("Fitness =", best_global_fit)

    return {sid: best_global[i] for i, sid in enumerate(switches)}
