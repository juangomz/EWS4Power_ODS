import mosaik

import importlib
import simuladores.wind_sim
importlib.reload(simuladores.wind_sim)


SIM_CONFIG = {
    'WindSim2D': {'python': 'simuladores.wind_sim:WindSim2D'},
    'FailureModel': {'python': 'simuladores.failure_model:FailureModel'},
    'PyPSA_Sim': {'python': 'simuladores.pypsa_sim:PyPSASim'},
}

def main():
    world = mosaik.World(SIM_CONFIG)

    wind = world.start('WindSim2D', time_resolution=3600)
    failure = world.start('FailureModel', time_resolution=3600)
    grid = world.start('PyPSA_Sim', time_resolution=3600)

    w = wind.WindSim2D.create(1)[0]
    g = grid.PyPSA_Grid.create(1)[0]

    # Cantidad de líneas a simular (puedes leerla de tu red PyPSA si lo deseas)
    # Obtener line_positions del grid antes de la simulación
    grid_data = world.get_data({g: ['num_lines', 'line_positions']})
    num_lines = list(grid_data.values())[0]['num_lines']
    line_positions = list(grid_data.values())[0]['line_positions']

    # 💡 Pasa line_positions como parámetro
    failures = failure.FailureModel.create(num_lines, line_positions=line_positions)
        
    # --- Conexiones ---
    # 1️⃣ El viento alimenta a todos los modelos de fallo
    for f in failures:
        world.connect(w, f, 'wind_speed', 'grid_lon', 'grid_lat', 'wind_shape')

    
    # 2️⃣ Cada modelo de fallo controla la red
    for f in failures:
        world.connect(f, g, ('line_status', 'line_status'))

    # 3️⃣ El viento también alimenta al grid directamente
    world.connect(w, g, ('wind_speed', 'wind_speed'))
    
    # Ejecutar simulación por 24 horas
    world.run(until=24 * 3600)

if __name__ == '__main__':
    main()
