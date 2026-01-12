import mosaik

SIM_CONFIG = {
    'ClimateModel': {'python': 'simuladores.Climate_model:ClimateModel'},
    'FailureModel': {'python': 'simuladores.Failure_model:FailureModel'},
    'PPModel': {'python': 'simuladores.PP_model:PPModel'},
    'OpDecisionModel': {'python': 'simuladores.Op_Decision_model:OpDecisionModel'}
}

FORECAST_HORIZON = 0

def main():
    world = mosaik.World(SIM_CONFIG)

    # --- Inicialización de simuladores ---
    print("⚙️ Iniciando simuladores...")

    climate = world.start('ClimateModel', step_size=3600)
    failure = world.start('FailureModel', step_size=3600)
    decision = world.start('OpDecisionModel', step_size=3600)
    grid = world.start('PPModel', step_size=3600)

    # --- Crear entidades ---
    c = climate.ClimateModel.create(1, forecast_horizon=FORECAST_HORIZON)[0]
    g = grid.PPModel.create(1)[0]

    # --- Obtener posiciones de líneas del grid ---
    grid_data = world.get_data({g: ['line_positions', 'line_status', 'lines', 'buses', 'transformers', 'loads']})
    line_positions = list(grid_data.values())[0]['line_positions']
    line_status = list(grid_data.values())[0]['line_status']
    lines = list(grid_data.values())[0]['lines']
    buses = list(grid_data.values())[0]['buses']
    switches = list(grid_data.values())[0]['switches']
    transformers = list(grid_data.values())[0]['transformers']
    loads = list(grid_data.values())[0]['loads']

    # --- Crear entidades dependientes ---
    f = failure.FailureModel.create(1, line_positions=line_positions)[0]
    d = decision.OpDecisionModel.create(1)[0]
    
    # ================================================================
    # CONEXIONES ENTRE SIMULADORES
    # ================================================================

    # 🌀 Clima → Fallo
    world.connect(c, f, 'climate', 'grid_x', 'grid_y', 'shape')
    
    # El viento también alimenta al grid directamente
    world.connect(c, g, 'climate', 'grid_x', 'grid_y', 'shape')

    # 🌀 Fallo → Decisión (probabilidades)
    world.connect(f, d, 'fail_prob', 'fail_prob')

    # 🌀 Fallo → Red (probabilidades)
    world.connect(f, g, 'fail_prob','fail_prob')

    # 🧩 Decisión → Red (plan de reparación)
    world.connect(d, g, 'repair_plan', 'switch_plan')

    # 🔁 Red → Decisión (estado actualizado)
    world.connect(g, d, 'line_status', 'lines', 'buses', 'switches', 'transformers', 'loads', time_shifted=True, initial_data={'line_status':line_status, 'lines':lines, 'buses':buses, 'switches':switches, 'transformers':transformers, 'loads':loads})
    
    # Ejecutar simulación por 24 horas
    world.run(until=24 * 3600)

from pyinstrument import Profiler

if __name__ == "__main__":
    profiler = Profiler(interval=0.001, async_mode="enabled")
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