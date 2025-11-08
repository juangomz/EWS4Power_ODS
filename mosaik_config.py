import mosaik

import importlib
import simuladores.Climate_model
importlib.reload(simuladores.Climate_model)


SIM_CONFIG = {
    'ClimateModel': {'python': 'simuladores.Climate_model:ClimateModel'},
    'FailureModel': {'python': 'simuladores.Failure_model:FailureModel'},
    'PPModel': {'python': 'simuladores.PP_model:PPModel'},
    'OpDecisionModel': {'python': 'simuladores.Op_Decision_model:OpDecisionModel'}
    # 'PyPSA_Sim': {'python': 'simuladores.pypsa_sim:PyPSASim'},
}

def main():
    world = mosaik.World(SIM_CONFIG)

    climate = world.start('ClimateModel', time_resolution=3600)
    failure = world.start('FailureModel', time_resolution=3600)
    decision = world.start('OpDecisionModel', time_resolution=3600)
    grid = world.start('PPModel', time_resolution=3600)
    

    c = climate.ClimateModel.create(1)[0]
    g = grid.PPModel.create(1)[0]


    # Obtener line_positions del grid antes de la simulación
    grid_data = world.get_data({g: ['num_lines', 'line_positions']})
    num_lines = list(grid_data.values())[0]['num_lines']
    line_positions = list(grid_data.values())[0]['line_positions']

    # Pasa line_positions como parámetro
    f = failure.FailureModel.create(1, line_positions=line_positions)[0]
    d = decision.OpDecisionModel.create(1)[0]
    # --- Conexiones ---
    # El viento alimenta a todos los modelos de fallo
    world.connect(c, f, 'wind_speed', 'grid_x', 'grid_y', 'wind_shape')
    
    # Cada modelo de fallo controla la red
    # for f in failures:
    #     world.connect(f, g, ('line_status', 'line_status'))
    world.connect(f, d, 'line_status', 'line_status')
        
    world.connect(d, g, 'line_status', 'line_status')

    # El viento también alimenta al grid directamente
    world.connect(c, g, 'wind_speed', 'grid_x', 'grid_y', 'wind_shape')
    
    # Ejecutar simulación por 24 horas
    world.run(until=24 * 3600)

from pyinstrument import Profiler

if __name__ == "__main__":
    profiler = Profiler(interval=0.001, async_mode=True)
    profiler.start()
    print("🚀 Simulación iniciada (usa Ctrl+C para detener y ver el perfil)...")

    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Simulación interrumpida manualmente.")
    finally:
        profiler.stop()
        with open("pyinstrument_report.html", "w", encoding="utf-8") as f:
            f.write(profiler.output_html())
        print("✅ Informe guardado en pyinstrument_report.html")