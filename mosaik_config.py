import mosaik

import importlib
import simuladores.wind_sim
importlib.reload(simuladores.wind_sim)


SIM_CONFIG = {
    'WindSim': {'python': 'simuladores.wind_sim:WindSim'},
    'FailureModel': {'python': 'simuladores.failure_model:FailureModel'},
    'PyPSA_Sim': {'python': 'simuladores.pypsa_sim:PyPSASim'},
}

def main():
    world = mosaik.World(SIM_CONFIG)

    wind = world.start('WindSim', time_resolution=3600)
    failure = world.start('FailureModel', time_resolution=3600)
    grid = world.start('PyPSA_Sim', time_resolution=3600)

    w = wind.WindSim.create(1)[0]
    g = grid.PyPSA_Grid.create(1)[0]

    # Cantidad de líneas a simular (puedes leerla de tu red PyPSA si lo deseas)
    grid_data = world.get_data({g: ['num_lines', 'ens', 'current']})
    print("GRID DATA:", grid_data)
    num_lines = list(grid_data.values())[0]["num_lines"]    # Crear N modelos de fallo, uno por línea
    
    failures = failure.FailureModel.create(num_lines)
    
    # --- Conexiones ---
    # 1️⃣ El viento alimenta a todos los modelos de fallo
    for f in failures:
        world.connect(w, f, ('wind_speed', 'wind_speed'))

    # 2️⃣ Cada modelo de fallo controla la red
    for f in failures:
        world.connect(f, g, ('line_status', 'line_status'))

    # 3️⃣ El viento también alimenta al grid directamente
    world.connect(w, g, ('wind_speed', 'wind_speed'))
    
    # Ejecutar simulación por 24 horas
    world.run(until=24 * 3600)

if __name__ == '__main__':
    main()
