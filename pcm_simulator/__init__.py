"""
PCM Simulator Package
=====================
Streamlit application for PCM heat exchanger simulation.
"""

from .pcm_library import (
    PCMCategory,
    PCMMaterial,
    EnthalpyData,
    PCMMaterialLibrary,
    get_library
)

from .geometry import (
    BoxGeometry,
    HEXGeometry,
    GeometryPresets
)

from .simulation_core import (
    Phase,
    SimulationMode,
    WaterProperties,
    OperatingConditions,
    SimulationConfig,
    SimulationState,
    SimulationResults,
    SimulationSummary,
    PCMHeatExchangerSimulation
)

from .visualization import (
    create_temperature_plot,
    create_power_plot,
    create_energy_plot,
    create_melt_fraction_plot,
    create_front_position_plot,
    create_enthalpy_plot,
    create_combined_dashboard,
    create_enthalpy_curve_plot,
    export_data_to_csv
)

__version__ = "1.0.0"
__all__ = [
    # PCM Library
    'PCMCategory',
    'PCMMaterial',
    'EnthalpyData',
    'PCMMaterialLibrary',
    'get_library',
    # Geometry
    'BoxGeometry',
    'HEXGeometry',
    'GeometryPresets',
    # Simulation
    'Phase',
    'SimulationMode',
    'WaterProperties',
    'OperatingConditions',
    'SimulationConfig',
    'SimulationState',
    'SimulationResults',
    'SimulationSummary',
    'PCMHeatExchangerSimulation',
    # Visualization
    'create_temperature_plot',
    'create_power_plot',
    'create_energy_plot',
    'create_melt_fraction_plot',
    'create_front_position_plot',
    'create_enthalpy_plot',
    'create_combined_dashboard',
    'create_enthalpy_curve_plot',
    'export_data_to_csv',
]
