"""
Simulation Core Module
======================
PCM Heat Exchanger simulation engine with moving boundary model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum

try:
    from .pcm_library import PCMMaterial, PCMCategory, EnthalpyData
    from .geometry import BoxGeometry, HEXGeometry
except ImportError:
    from pcm_library import PCMMaterial, PCMCategory, EnthalpyData
    from geometry import BoxGeometry, HEXGeometry


# =============================================================================
# ENUMERATIONS
# =============================================================================

class Phase(Enum):
    """PCM phase states"""
    FULLY_SOLID = "Fully Solid"
    MELTING = "Melting"
    FULLY_LIQUID = "Fully Liquid"
    SOLIDIFYING = "Solidifying"
    SUPERCOOLED = "Supercooled"


class SimulationMode(Enum):
    """Operating modes"""
    CHARGING = "Charging"
    DISCHARGING = "Discharging"
    IDLE = "Idle"


class WaterSupplyMode(Enum):
    """Water supply modes"""
    CONSTANT_TEMPERATURE = "Constant Temperature"
    HEAT_SOURCE_SINK = "Heat Source/Sink"


class SourceControlMode(Enum):
    """Heat source/sink control modes"""
    CONSTANT_POWER = "Constant Power"
    THERMOSTAT = "Thermostat"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WaterProperties:
    """Water properties"""
    rho: float = 1000        # Density [kg/m3]
    mu: float = 0.000682     # Dynamic viscosity [Pa*s]
    k: float = 0.6           # Thermal conductivity [W/m*K]
    cp: float = 4180         # Specific heat [J/kg*K]
    Pr: float = 4.53         # Prandtl number [-]


@dataclass
class HeatSourceSinkConfig:
    """Configuration for heat source/sink closed-loop mode"""
    power_kW: float = 5.0                                       # Always positive; sign auto-determined
    tank_volume_L: float = 50.0                                  # Tank volume [L]
    T_tank_initial: float = 25.0                                 # Initial tank temperature [C]
    control_mode: SourceControlMode = SourceControlMode.CONSTANT_POWER
    T_setpoint: float = 60.0                                     # Thermostat setpoint [C]


@dataclass
class OperatingConditions:
    """Operating conditions"""
    Q_water_lpm: float       # Water flow rate [lpm]
    T_water_hot: float       # Hot water temperature [C]
    T_water_cold: float      # Cold water temperature [C]
    water_supply_mode: WaterSupplyMode = WaterSupplyMode.CONSTANT_TEMPERATURE
    heat_source_config: Optional[HeatSourceSinkConfig] = None


@dataclass
class WallLossConfig:
    """Configuration for tank wall heat loss to ambient"""
    enabled: bool = False
    T_ambient: float = 25.0            # [°C]
    wall_thickness_mm: float = 2.0     # [mm]
    k_wall: float = 50.0               # Wall conductivity [W/(m·K)]
    h_ext: float = 10.0                # External convection [W/(m²·K)]


@dataclass
class PipeLossConfig:
    """Configuration for pipe heat losses between components (closed-loop mode only)"""
    enabled: bool = False
    T_ambient: float = 25.0                  # Ambient temperature [°C]
    pipe_OD_mm: float = 30.0                 # Pipe outer diameter [mm]
    insulation_thickness_mm: float = 13.0    # Insulation thickness [mm] (0 = bare pipe)
    k_insulation: float = 0.04               # Insulation conductivity [W/(m·K)]
    h_ext: float = 10.0                      # External convection coefficient [W/(m²·K)]
    L_tank_to_source_m: float = 20.0         # Pipe 1: Tank → Source/Sink [m]
    L_source_to_hex_m: float = 20.0          # Pipe 2: Source/Sink → HEX Cell [m]
    L_hex_to_tank_m: float = 5.0             # Pipe 3: HEX Cell → Tank [m]


@dataclass
class SimulationConfig:
    """Simulation configuration"""
    dt: float = 1.0                    # Time step [seconds]
    t_max: float = 7200                # Maximum simulation time [seconds]
    T_pcm_initial: float = 40.0        # Initial PCM temperature [C]
    convergence_tol: float = 0.001     # Convergence tolerance
    max_iterations: int = 50           # Maximum iterations
    supercooling_deg: float = 0.0      # Degrees below T_liquidus before nucleation
    wall_loss: Optional[WallLossConfig] = None
    pipe_loss: Optional[PipeLossConfig] = None


@dataclass
class SimulationState:
    """Current state of the simulation"""
    time: float = 0.0                  # Current time [s]

    # Enthalpy tracking
    H_pcm_specific: float = 0.0        # Specific enthalpy [kJ/kg]
    H_pcm_total: float = 0.0           # Total enthalpy [kJ]

    # Temperature
    T_pcm: float = 40.0                # Average PCM temperature [C]
    T_water_out: float = 40.0          # Water outlet temperature [C]

    # Front positions
    delta_fin: float = 0.0             # Front position from fin [m]
    r_front_tube: float = 0.0          # Front position from tube [m]

    # Fractions
    f_melted_fin: float = 0.0          # Melt fraction from fin path
    f_melted_tube: float = 0.0         # Melt fraction from tube path
    f_melted_total: float = 0.0        # Total melt fraction

    # Heat transfer
    Q_total: float = 0.0               # Total heat transfer rate [W]
    Q_fin: float = 0.0                 # Heat through fin path [W]
    Q_tube: float = 0.0                # Heat through tube path [W]

    # Cumulative energy
    E_total: float = 0.0               # Cumulative energy [kJ]

    # Wall heat loss
    Q_loss: float = 0.0                # Wall heat loss rate [W]
    E_loss: float = 0.0                # Cumulative wall loss energy [kJ]

    # Pipe heat loss (closed-loop mode)
    Q_pipe_loss: float = 0.0           # Total pipe loss rate [W]
    E_pipe_loss: float = 0.0           # Cumulative pipe loss energy [kJ]

    # Phase and mode
    phase: Phase = Phase.FULLY_SOLID
    mode: SimulationMode = SimulationMode.IDLE
    curve: str = 'melting'             # Which enthalpy curve to use

    # Closed-loop water mode
    T_water_in: float = 0.0    # Water inlet temp to HEX (varies in closed-loop mode)
    T_tank: float = 0.0        # Tank temperature
    Q_source: float = 0.0      # Heat source/sink power this step [W] (signed)

    # Loop temperature sensors (closed-loop mode)
    T_at_source_inlet: float = 0.0   # After Pipe 1 loss, before source
    T_after_source: float = 0.0      # After source, before Pipe 2 loss
    T_return_to_tank: float = 0.0    # After Pipe 3 loss, entering tank

    # Per-segment pipe losses
    Q_pipe_loss_1: float = 0.0       # Pipe 1: Tank → Source [W]
    Q_pipe_loss_2: float = 0.0       # Pipe 2: Source → HEX [W]
    Q_pipe_loss_3: float = 0.0       # Pipe 3: HEX → Tank [W]

    # Supercooling tracking
    is_supercooled: bool = False
    H_supercool_start: float = 0.0    # Enthalpy when supercooling began [kJ/kg]
    T_supercool_start: float = 0.0    # Temperature when supercooling began [C]


@dataclass
class SimulationResults:
    """Storage for simulation results"""
    time: List[float] = field(default_factory=list)
    T_pcm: List[float] = field(default_factory=list)
    T_water_out: List[float] = field(default_factory=list)
    Q_total: List[float] = field(default_factory=list)
    Q_fin: List[float] = field(default_factory=list)
    Q_tube: List[float] = field(default_factory=list)
    E_total: List[float] = field(default_factory=list)
    f_melted: List[float] = field(default_factory=list)
    delta_fin: List[float] = field(default_factory=list)
    r_front_tube: List[float] = field(default_factory=list)
    phase: List[str] = field(default_factory=list)
    H_specific: List[float] = field(default_factory=list)
    T_water_in: List[float] = field(default_factory=list)
    T_tank: List[float] = field(default_factory=list)
    Q_source_kW: List[float] = field(default_factory=list)
    Q_loss: List[float] = field(default_factory=list)
    E_loss: List[float] = field(default_factory=list)
    Q_pipe_loss: List[float] = field(default_factory=list)
    E_pipe_loss: List[float] = field(default_factory=list)

    # Loop temperature sensors (closed-loop mode)
    T_at_source_inlet: List[float] = field(default_factory=list)
    T_after_source: List[float] = field(default_factory=list)
    T_return_to_tank: List[float] = field(default_factory=list)

    # Per-segment pipe losses
    Q_pipe_loss_1: List[float] = field(default_factory=list)
    Q_pipe_loss_2: List[float] = field(default_factory=list)
    Q_pipe_loss_3: List[float] = field(default_factory=list)

    def record(self, state: SimulationState):
        """Record current state"""
        self.time.append(state.time)
        self.T_pcm.append(state.T_pcm)
        self.T_water_out.append(state.T_water_out)
        self.Q_total.append(state.Q_total / 1000)  # Convert to kW
        self.Q_fin.append(state.Q_fin / 1000)
        self.Q_tube.append(state.Q_tube / 1000)
        self.E_total.append(state.E_total / 3600)  # Convert to kWh
        self.f_melted.append(state.f_melted_total)
        self.delta_fin.append(state.delta_fin * 1000)  # Convert to mm
        self.r_front_tube.append(state.r_front_tube * 1000)  # Convert to mm
        self.phase.append(state.phase.value)
        self.H_specific.append(state.H_pcm_specific)
        self.T_water_in.append(state.T_water_in)
        self.T_tank.append(state.T_tank)
        self.Q_source_kW.append(state.Q_source / 1000)
        self.Q_loss.append(state.Q_loss / 1000)       # Convert to kW
        self.E_loss.append(state.E_loss / 3600)        # Convert to kWh
        self.Q_pipe_loss.append(state.Q_pipe_loss / 1000)  # Convert to kW
        self.E_pipe_loss.append(state.E_pipe_loss / 3600)  # Convert to kWh
        self.T_at_source_inlet.append(state.T_at_source_inlet)
        self.T_after_source.append(state.T_after_source)
        self.T_return_to_tank.append(state.T_return_to_tank)
        self.Q_pipe_loss_1.append(state.Q_pipe_loss_1 / 1000)  # Convert to kW
        self.Q_pipe_loss_2.append(state.Q_pipe_loss_2 / 1000)  # Convert to kW
        self.Q_pipe_loss_3.append(state.Q_pipe_loss_3 / 1000)  # Convert to kW

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert to numpy arrays"""
        return {
            'time_s': np.array(self.time),
            'time_min': np.array(self.time) / 60,
            'T_pcm_C': np.array(self.T_pcm),
            'T_water_out_C': np.array(self.T_water_out),
            'Q_total_kW': np.array(self.Q_total),
            'Q_fin_kW': np.array(self.Q_fin),
            'Q_tube_kW': np.array(self.Q_tube),
            'E_total_kWh': np.array(self.E_total),
            'f_melted': np.array(self.f_melted),
            'delta_fin_mm': np.array(self.delta_fin),
            'r_front_tube_mm': np.array(self.r_front_tube),
            'H_specific_kJ_kg': np.array(self.H_specific),
            'T_water_in_C': np.array(self.T_water_in),
            'T_tank_C': np.array(self.T_tank),
            'Q_source_kW': np.array(self.Q_source_kW),
            'Q_loss_kW': np.array(self.Q_loss),
            'E_loss_kWh': np.array(self.E_loss),
            'Q_pipe_loss_kW': np.array(self.Q_pipe_loss),
            'E_pipe_loss_kWh': np.array(self.E_pipe_loss),
            'T_at_source_inlet_C': np.array(self.T_at_source_inlet),
            'T_after_source_C': np.array(self.T_after_source),
            'T_return_to_tank_C': np.array(self.T_return_to_tank),
            'Q_pipe_loss_1_kW': np.array(self.Q_pipe_loss_1),
            'Q_pipe_loss_2_kW': np.array(self.Q_pipe_loss_2),
            'Q_pipe_loss_3_kW': np.array(self.Q_pipe_loss_3),
        }

    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        import pandas as pd
        data = self.to_numpy()
        df = pd.DataFrame({
            'Time (s)': data['time_s'],
            'Time (min)': data['time_min'],
            'T_PCM (C)': data['T_pcm_C'],
            'T_water_out (C)': data['T_water_out_C'],
            'Q_total (kW)': data['Q_total_kW'],
            'Q_fin (kW)': data['Q_fin_kW'],
            'Q_tube (kW)': data['Q_tube_kW'],
            'E_total (kWh)': data['E_total_kWh'],
            'Melt Fraction': data['f_melted'],
            'Delta_fin (mm)': data['delta_fin_mm'],
            'R_front_tube (mm)': data['r_front_tube_mm'],
            'H_specific (kJ/kg)': data['H_specific_kJ_kg'],
            'T_water_in (C)': data['T_water_in_C'],
            'T_tank (C)': data['T_tank_C'],
            'Q_source (kW)': data['Q_source_kW'],
            'Q_loss (kW)': data['Q_loss_kW'],
            'E_loss (kWh)': data['E_loss_kWh'],
            'Q_pipe_loss (kW)': data['Q_pipe_loss_kW'],
            'E_pipe_loss (kWh)': data['E_pipe_loss_kWh'],
            'T_at_source_inlet (C)': data['T_at_source_inlet_C'],
            'T_after_source (C)': data['T_after_source_C'],
            'T_return_to_tank (C)': data['T_return_to_tank_C'],
            'Q_pipe_loss_1 (kW)': data['Q_pipe_loss_1_kW'],
            'Q_pipe_loss_2 (kW)': data['Q_pipe_loss_2_kW'],
            'Q_pipe_loss_3 (kW)': data['Q_pipe_loss_3_kW'],
        })
        return df


@dataclass
class SimulationSummary:
    """Summary of simulation results"""
    total_time_s: float
    total_time_min: float
    final_T_pcm: float
    total_energy_kWh: float
    average_power_kW: float
    final_melt_fraction: float
    final_phase: str
    initial_T_pcm: float
    pcm_mass_kg: float
    mode: str


# =============================================================================
# MAIN SIMULATION CLASS
# =============================================================================

class PCMHeatExchangerSimulation:
    """
    Time-based simulation of PCM heat exchanger with moving front model
    """

    def __init__(
        self,
        hex_geometry: HEXGeometry,
        box_geometry: BoxGeometry,
        pcm: PCMMaterial,
        operating: OperatingConditions,
        config: SimulationConfig,
        water: Optional[WaterProperties] = None,
        progress_callback: Optional[Callable[[float, SimulationState], None]] = None
    ):
        self.hex = hex_geometry
        self.box = box_geometry
        self.pcm = pcm
        self.water = water or WaterProperties()
        self.operating = operating
        self.config = config
        self.progress_callback = progress_callback

        # Calculate derived quantities
        self._calculate_derived_quantities()

        # Tank mass for closed-loop mode
        if self.operating.water_supply_mode == WaterSupplyMode.HEAT_SOURCE_SINK:
            cfg = self.operating.heat_source_config
            self.m_tank = cfg.tank_volume_L * self.water.rho / 1000  # kg

        # Initialize state
        self.state = SimulationState()
        self.results = SimulationResults()

        # Initialize state
        self._initialize_state()

    def _calculate_derived_quantities(self):
        """Calculate quantities derived from inputs"""
        # PCM volume and mass (use liquid density)
        hex_volume = self.hex.estimate_hex_volume()
        self.V_pcm = self.box.volume_m3 - hex_volume  # m3
        self.m_pcm = self.V_pcm * self.pcm.rho_liquid  # kg

        # Water mass flow rate
        self.m_dot_water = self.operating.Q_water_lpm / 60 * self.water.rho / 1000  # kg/s

        # Water heat capacity rate
        self.C_water = self.m_dot_water * self.water.cp  # W/K

        # Water-side heat transfer coefficient
        D_in_m = self.hex.D_in / 1000
        A_flow_m2 = self.hex.A_flow_m2
        V_water = self.m_dot_water / (self.water.rho * A_flow_m2)
        Re_water = self.water.rho * V_water * D_in_m / self.water.mu

        if Re_water < 2300:
            Nu_water = 3.66
        else:
            Nu_water = 0.023 * (Re_water ** 0.8) * (self.water.Pr ** 0.4)

        self.h_water = Nu_water * self.water.k / D_in_m
        self.Re_water = Re_water  # Store for reference

        # Thermal resistances (per unit area)
        self.R_water = 1 / self.h_water  # m2K/W
        self.R_wall = self.hex.r_in * np.log(self.hex.r_out / self.hex.r_in) / self.hex.k_tube  # m2K/W

        # Pressure drop calculation (Blasius friction factor)
        self._calculate_pressure_drop(V_water, D_in_m, Re_water)

        # Fin efficiency parameters
        self._calculate_fin_parameters()

        # Volume fractions for fin vs tube paths
        self._calculate_volume_fractions()

        # Wall heat loss UA
        self._wall_loss_UA = 0.0
        if self.config.wall_loss and self.config.wall_loss.enabled:
            wl = self.config.wall_loss
            R = (wl.wall_thickness_mm / 1000) / wl.k_wall + 1.0 / wl.h_ext
            self._wall_loss_UA = self.box.surface_area_m2 / R   # W/K
            self._T_ambient = wl.T_ambient
        else:
            self._T_ambient = 25.0

        # Pipe heat loss UA (closed-loop mode only)
        self._pipe_UA_1 = 0.0
        self._pipe_UA_2 = 0.0
        self._pipe_UA_3 = 0.0
        self._pipe_T_ambient = 25.0
        if self.config.pipe_loss and self.config.pipe_loss.enabled:
            pl = self.config.pipe_loss
            self._pipe_T_ambient = pl.T_ambient
            r_pipe = (pl.pipe_OD_mm / 2) / 1000           # pipe outer radius [m]
            r_outer = r_pipe + pl.insulation_thickness_mm / 1000  # insulation outer radius [m]
            for attr, L in [('_pipe_UA_1', pl.L_tank_to_source_m),
                            ('_pipe_UA_2', pl.L_source_to_hex_m),
                            ('_pipe_UA_3', pl.L_hex_to_tank_m)]:
                if L <= 0:
                    setattr(self, attr, 0.0)
                    continue
                R_ins = 0.0
                if pl.insulation_thickness_mm > 0 and pl.k_insulation > 0:
                    R_ins = np.log(r_outer / r_pipe) / (2 * np.pi * pl.k_insulation * L)
                R_ext = 1.0 / (2 * np.pi * r_outer * L * pl.h_ext)
                setattr(self, attr, 1.0 / (R_ins + R_ext))

    def _calculate_fin_parameters(self):
        """Calculate fin efficiency parameters"""
        # Fin geometry
        self.r_eq = self.hex.tube_pitch / 2 / 1000  # Equivalent fin radius [m]
        self.L_fin = self.r_eq - self.hex.r_out  # Fin length [m]
        self.t_fin_m = self.hex.t_fin / 1000  # Fin thickness [m]

        # Radius ratio for annular fin correction
        self.radius_ratio = self.r_eq / self.hex.r_out
        self.phi_fin = (self.radius_ratio - 1) * (1 + 0.35 * np.log(self.radius_ratio))

    def _calculate_volume_fractions(self):
        """Calculate what fraction of PCM volume is associated with fin vs tube paths"""
        self.f_vol_fin = self.hex.A_secondary / self.hex.A_total
        self.f_vol_tube = self.hex.A_primary / self.hex.A_total

    def _calculate_pressure_drop(self, V_water: float, D_in_m: float, Re_water: float):
        """
        Calculate water-side pressure drop using Blasius friction factor.

        For turbulent flow (Re > 2300): f = 0.316 * Re^(-0.25)
        For laminar flow (Re < 2300): f = 64 / Re

        Pressure drop: ΔP = f * (L/D) * ρ * V² / 2

        Args:
            V_water: Water velocity [m/s]
            D_in_m: Tube inner diameter [m]
            Re_water: Reynolds number [-]
        """
        # Total tube length per flow path [m]
        L_tube_total_m = self.hex.L_tube_total / 1000

        # Friction factor
        if Re_water < 2300:
            # Laminar: Hagen-Poiseuille
            f = 64 / Re_water if Re_water > 0 else 0
        else:
            # Turbulent: Blasius correlation (valid for smooth pipes, Re < 100,000)
            f = 0.316 * (Re_water ** (-0.25))

        # Pressure drop [Pa]
        # ΔP = f * (L/D) * ρ * V² / 2
        self.delta_P_Pa = f * (L_tube_total_m / D_in_m) * self.water.rho * (V_water ** 2) / 2
        self.delta_P_kPa = self.delta_P_Pa / 1000
        self.delta_P_bar = self.delta_P_Pa / 100000
        self.friction_factor = f

    def _initialize_state(self):
        """Initialize simulation state based on initial temperature"""
        T_init = self.config.T_pcm_initial

        # Phase change temperature range
        T_solidus = self.pcm.T_solidus
        T_liquidus = self.pcm.T_liquidus

        self.state.T_pcm = T_init
        self.state.T_water_out = T_init
        self.state.time = 0.0
        self.state.E_total = 0.0

        # Determine initial phase and front positions
        if T_init <= T_solidus:
            self.state.phase = Phase.FULLY_SOLID
            self.state.curve = 'melting'
            self.state.delta_fin = 0.0
            self.state.r_front_tube = self.hex.r_out
            self.state.f_melted_total = 0.0
            self.state.f_melted_fin = 0.0
            self.state.f_melted_tube = 0.0
        elif T_init >= T_liquidus:
            self.state.phase = Phase.FULLY_LIQUID
            self.state.curve = 'solidifying'
            self.state.delta_fin = self.hex.delta_max_fin
            self.state.r_front_tube = self.hex.r_max_tube
            self.state.f_melted_total = 1.0
            self.state.f_melted_fin = 1.0
            self.state.f_melted_tube = 1.0
        else:
            f_approx = (T_init - T_solidus) / (T_liquidus - T_solidus)
            self.state.phase = Phase.MELTING
            self.state.curve = 'melting'
            self.state.delta_fin = f_approx * self.hex.delta_max_fin
            self.state.r_front_tube = self.hex.r_out + f_approx * (self.hex.r_max_tube - self.hex.r_out)
            self.state.f_melted_total = f_approx
            self.state.f_melted_fin = f_approx
            self.state.f_melted_tube = f_approx

        # Calculate initial enthalpy
        self.state.H_pcm_specific = self.pcm.enthalpy_data.get_cumulative_H(T_init, self.state.curve)
        self.state.H_pcm_total = self.state.H_pcm_specific * self.m_pcm

        # Initialize closed-loop water mode fields
        if self.operating.water_supply_mode == WaterSupplyMode.HEAT_SOURCE_SINK:
            cfg = self.operating.heat_source_config
            self.state.T_tank = cfg.T_tank_initial
            self.state.T_water_in = cfg.T_tank_initial
        else:
            self.state.T_water_in = T_init

    def get_system_summary(self) -> Dict:
        """Get system summary as dictionary"""
        hex_vol = self.hex.estimate_hex_volume()
        E_latent = self.m_pcm * self.pcm.enthalpy_data.total_latent_heat_melting / 3600  # kWh
        E_sensible = self.m_pcm * self.pcm.cp_sensible * (self.pcm.T_liquidus - self.pcm.T_solidus) / 3600  # kWh

        return {
            "Box Volume (L)": f"{self.box.volume_liters:.1f}",
            "HEX Volume (L)": f"{hex_vol*1000:.2f}",
            "PCM Volume (L)": f"{self.V_pcm * 1000:.1f}",
            "PCM Mass (kg)": f"{self.m_pcm:.1f}",
            "Total HT Area (m2)": f"{self.hex.A_total:.2f}",
            "Latent Capacity (kWh)": f"{E_latent:.2f}",
            "Sensible Capacity (kWh)": f"{E_sensible:.2f}",
            "Total Capacity (kWh)": f"{E_latent + E_sensible:.2f}",
            "Water Flow (lpm)": f"{self.operating.Q_water_lpm}",
            "h_water (W/m2K)": f"{self.h_water:.1f}",
            "Re_water": f"{self.Re_water:.0f}",
            "Pressure Drop (kPa)": f"{self.delta_P_kPa:.1f}",
            "Pressure Drop (bar)": f"{self.delta_P_bar:.3f}",
        }

    def _get_natural_convection_factor(self, delta_liquid: float) -> float:
        """Natural convection factor - always returns 1.0 (pure conduction)"""
        return 1.0

    def _calculate_fin_efficiency(self, h_pcm: float) -> Tuple[float, float]:
        """Calculate fin efficiency and surface efficiency"""
        if h_pcm <= 0 or self.t_fin_m <= 0:
            return 1.0, 1.0

        m = np.sqrt(2 * h_pcm / (self.hex.k_fin * self.t_fin_m))
        mL_phi = m * self.L_fin * self.phi_fin

        if mL_phi < 0.01:
            eta_fin = 1.0
        else:
            eta_fin = np.tanh(mL_phi) / mL_phi

        A_ratio = self.hex.A_secondary / self.hex.A_total
        eta_o = 1 - A_ratio * (1 - eta_fin)

        return eta_fin, eta_o

    def _calculate_heat_transfer(self, T_water_in: float, T_pcm: float,
                                  delta_fin: float, r_front_tube: float) -> Tuple[float, float, float, float]:
        """Calculate heat transfer rates through fin and tube paths"""
        T_solidus = self.pcm.T_solidus
        T_liquidus = self.pcm.T_liquidus

        if T_pcm <= T_solidus:
            f_liquid = 0.0
        elif T_pcm >= T_liquidus:
            f_liquid = 1.0
        else:
            f_liquid = (T_pcm - T_solidus) / (T_liquidus - T_solidus)

        Nu_conv = self._get_natural_convection_factor(delta_fin)
        k_pcm_eff = self.pcm.get_k_effective(f_liquid, Nu_conv)

        delta_max = self.hex.delta_max_fin

        # FIN PATH
        if self.state.phase == Phase.FULLY_SOLID:
            R_pcm_fin = delta_max / self.pcm.k_solid
        elif self.state.phase in (Phase.FULLY_LIQUID, Phase.SUPERCOOLED):
            R_pcm_fin = delta_max / k_pcm_eff
        elif self.state.phase == Phase.MELTING:
            delta_liquid = max(delta_fin, 1e-9)
            delta_solid = max(delta_max - delta_fin, 1e-9)
            R_liquid = delta_liquid / k_pcm_eff
            R_solid = delta_solid / self.pcm.k_solid
            R_pcm_fin = R_liquid + R_solid
        else:  # SOLIDIFYING
            delta_solid = max(delta_max - delta_fin, 1e-9)
            delta_liquid = max(delta_fin, 1e-9)
            R_solid = delta_solid / self.pcm.k_solid
            R_liquid = delta_liquid / k_pcm_eff
            R_pcm_fin = R_solid + R_liquid

        h_pcm_eff = k_pcm_eff / delta_max
        eta_fin, eta_o = self._calculate_fin_efficiency(h_pcm_eff)

        R_total_fin = self.R_water + self.R_wall + R_pcm_fin / eta_o
        U_fin = 1 / R_total_fin if R_total_fin > 0 else 0

        # TUBE PATH
        r_out = self.hex.r_out
        r_max = self.hex.r_max_tube

        if self.state.phase == Phase.FULLY_SOLID:
            R_pcm_tube_raw = np.log(r_max / r_out) / (2 * np.pi * self.pcm.k_solid)
        elif self.state.phase in (Phase.FULLY_LIQUID, Phase.SUPERCOOLED):
            R_pcm_tube_raw = np.log(r_max / r_out) / (2 * np.pi * k_pcm_eff)
        elif self.state.phase == Phase.MELTING:
            r_front = max(r_front_tube, r_out + 1e-9)
            r_front = min(r_front, r_max - 1e-9)
            R_liquid_tube = np.log(r_front / r_out) / (2 * np.pi * k_pcm_eff)
            R_solid_tube = np.log(r_max / r_front) / (2 * np.pi * self.pcm.k_solid)
            R_pcm_tube_raw = R_liquid_tube + R_solid_tube
        else:  # SOLIDIFYING
            r_front = max(r_front_tube, r_out + 1e-9)
            r_front = min(r_front, r_max - 1e-9)
            R_solid_tube = np.log(r_front / r_out) / (2 * np.pi * self.pcm.k_solid)
            R_liquid_tube = np.log(r_max / r_front) / (2 * np.pi * k_pcm_eff)
            R_pcm_tube_raw = R_solid_tube + R_liquid_tube

        L_tube_total = (self.hex.L_tube_total / 1000) * self.hex.N_rows
        R_pcm_tube_area = R_pcm_tube_raw / L_tube_total if L_tube_total > 0 else 999999

        R_total_tube = self.R_water + self.R_wall + R_pcm_tube_area
        U_tube = 1 / R_total_tube if R_total_tube > 0 else 0

        # COMBINED HEAT TRANSFER
        UA_fin = U_fin * self.hex.A_secondary * eta_o
        UA_tube = U_tube * self.hex.A_primary
        UA_total = UA_fin + UA_tube

        NTU = UA_total / self.C_water if self.C_water > 0 else 0
        effectiveness = 1 - np.exp(-NTU) if NTU > 0 else 0

        Q_max = self.C_water * abs(T_water_in - T_pcm)
        Q_total = effectiveness * Q_max

        if UA_total > 0:
            Q_fin = Q_total * UA_fin / UA_total
            Q_tube = Q_total * UA_tube / UA_total
        else:
            Q_fin = Q_tube = 0

        if self.C_water > 0:
            T_water_out = T_water_in - Q_total / self.C_water * np.sign(T_water_in - T_pcm)
        else:
            T_water_out = T_pcm

        if T_water_in > T_pcm:
            T_water_out = max(T_pcm, min(T_water_in, T_water_out))
        else:
            T_water_out = min(T_pcm, max(T_water_in, T_water_out))

        return Q_total, Q_fin, Q_tube, T_water_out

    def _update_front_positions(self, Q_fin: float, Q_tube: float, dt: float):
        """Update front positions based on heat transfer"""
        dH_dT = self.pcm.enthalpy_data.get_dH_dT(self.state.T_pcm, self.state.curve)

        E_fin = Q_fin * dt
        E_tube = Q_tube * dt

        # FIN FRONT
        if self.state.phase in [Phase.MELTING, Phase.SOLIDIFYING]:
            h_eff = dH_dT * 1000

            if h_eff > 0 and self.hex.A_secondary > 0:
                d_delta_fin = E_fin / (self.pcm.rho_liquid * self.hex.A_secondary * h_eff)

                # Use phase to determine direction (not mode)
                # MELTING = front advances (delta increases)
                # SOLIDIFYING = front retreats (delta decreases)
                if self.state.phase == Phase.MELTING:
                    self.state.delta_fin += d_delta_fin
                    self.state.delta_fin = min(self.state.delta_fin, self.hex.delta_max_fin)
                else:  # SOLIDIFYING
                    self.state.delta_fin -= d_delta_fin
                    self.state.delta_fin = max(self.state.delta_fin, 0)

        # TUBE FRONT
        if self.state.phase in [Phase.MELTING, Phase.SOLIDIFYING]:
            h_eff = dH_dT * 1000

            if h_eff > 0:
                L_tube_total = (self.hex.L_tube_total / 1000) * self.hex.N_rows

                if self.state.r_front_tube > 0:
                    d_r = E_tube / (self.pcm.rho_liquid * 2 * np.pi * self.state.r_front_tube * L_tube_total * h_eff)

                    # Use phase to determine direction (not mode)
                    # MELTING = front advances (r increases)
                    # SOLIDIFYING = front retreats (r decreases)
                    if self.state.phase == Phase.MELTING:
                        self.state.r_front_tube += d_r
                        self.state.r_front_tube = min(self.state.r_front_tube, self.hex.r_max_tube)
                    else:  # SOLIDIFYING
                        self.state.r_front_tube -= d_r
                        self.state.r_front_tube = max(self.state.r_front_tube, self.hex.r_out)

        # Update melt fractions
        self.state.f_melted_fin = self.state.delta_fin / self.hex.delta_max_fin
        self.state.f_melted_tube = (self.state.r_front_tube**2 - self.hex.r_out**2) / (self.hex.r_max_tube**2 - self.hex.r_out**2)

        self.state.f_melted_total = (self.state.f_melted_fin * self.f_vol_fin +
                                     self.state.f_melted_tube * self.f_vol_tube)

    def _update_phase(self):
        """Update phase based on current state"""
        T = self.state.T_pcm
        f = self.state.f_melted_total

        T_solidus = self.pcm.T_solidus
        T_liquidus = self.pcm.T_liquidus

        if self.state.mode == SimulationMode.CHARGING:
            if self.pcm.category == PCMCategory.HOT:
                # Hot PCM charging = heating/melting
                if T < T_solidus:
                    self.state.phase = Phase.FULLY_SOLID
                elif T > T_liquidus or f >= 0.99:
                    self.state.phase = Phase.FULLY_LIQUID
                else:
                    self.state.phase = Phase.MELTING
            else:
                # Cold PCM charging = cooling/solidifying
                if T > T_liquidus:
                    self.state.phase = Phase.FULLY_LIQUID
                elif T < T_solidus or f <= 0.01:
                    self.state.phase = Phase.FULLY_SOLID
                else:
                    if self.config.supercooling_deg > 0 and not self.state.is_supercooled and self.state.phase == Phase.FULLY_LIQUID:
                        self.state.is_supercooled = True
                        self.state.phase = Phase.SUPERCOOLED
                        self.state.H_supercool_start = self.state.H_pcm_specific
                        self.state.T_supercool_start = self.state.T_pcm
                    elif self.state.is_supercooled:
                        self.state.phase = Phase.SUPERCOOLED
                    else:
                        self.state.phase = Phase.SOLIDIFYING
        else:  # DISCHARGING
            if self.pcm.category == PCMCategory.HOT:
                # Hot PCM discharging = cooling/solidifying
                if T > T_liquidus:
                    self.state.phase = Phase.FULLY_LIQUID
                elif T < T_solidus or f <= 0.01:
                    self.state.phase = Phase.FULLY_SOLID
                else:
                    if self.config.supercooling_deg > 0 and not self.state.is_supercooled and self.state.phase == Phase.FULLY_LIQUID:
                        self.state.is_supercooled = True
                        self.state.phase = Phase.SUPERCOOLED
                        self.state.H_supercool_start = self.state.H_pcm_specific
                        self.state.T_supercool_start = self.state.T_pcm
                    elif self.state.is_supercooled:
                        self.state.phase = Phase.SUPERCOOLED
                    else:
                        self.state.phase = Phase.SOLIDIFYING
            else:
                # Cold PCM discharging = heating/melting
                if T < T_solidus:
                    self.state.phase = Phase.FULLY_SOLID
                elif T > T_liquidus or f >= 0.99:
                    self.state.phase = Phase.FULLY_LIQUID
                else:
                    self.state.phase = Phase.MELTING

    def _apply_pipe_loss(self, T_fluid: float, UA: float) -> Tuple[float, float]:
        """
        Apply pipe heat loss to flowing fluid.

        Returns:
            (T_after_loss, Q_loss_W): Adjusted temperature and heat loss rate [W] (positive = heat lost)
        """
        if UA <= 0 or self.C_water <= 0:
            return T_fluid, 0.0
        Q_loss = UA * (T_fluid - self._pipe_T_ambient)   # W (positive when T_fluid > T_ambient)
        dT = Q_loss / self.C_water
        T_after = T_fluid - dT
        return T_after, Q_loss

    def _calculate_closed_loop_water_in(self, mode: SimulationMode) -> Tuple[float, float, float, float]:
        """
        Calculate water inlet temperature for heat source/sink closed-loop mode.

        Returns:
            (T_after_source, Q_source_W, Q_pipe1_W, T_at_source_inlet):
            Water temp after source/sink, signed source power [W],
            Pipe 1 heat loss [W], and temperature arriving at source inlet
        """
        cfg = self.operating.heat_source_config

        # --- Pipe 1: Tank → Source/Sink (loss to ambient) ---
        T_at_source_inlet, Q_pipe1 = self._apply_pipe_loss(self.state.T_tank, self._pipe_UA_1)

        # Determine sign: +1 = heat source (heating water), -1 = heat sink (cooling water)
        if mode == SimulationMode.CHARGING:
            if self.pcm.category == PCMCategory.HOT:
                sign = 1   # Charging hot PCM needs hot water → heat source
            else:
                sign = -1  # Charging cold PCM needs cold water → heat sink
        else:  # DISCHARGING
            if self.pcm.category == PCMCategory.HOT:
                sign = -1  # Discharging hot PCM needs cold water → heat sink
            else:
                sign = 1   # Discharging cold PCM needs warm water → heat source

        Q_source_W = sign * cfg.power_kW * 1000

        # Thermostat: proportional VFD control — modulate power to reach T_setpoint
        if cfg.control_mode == SourceControlMode.THERMOSTAT:
            Q_needed = self.C_water * (cfg.T_setpoint - self.state.T_tank)
            Q_max = cfg.power_kW * 1000
            Q_source_W = max(-Q_max, min(Q_max, Q_needed))

        # Clamp: a sink can only remove heat (Q <= 0), a source can only add (Q >= 0)
        if sign == -1:
            Q_source_W = min(Q_source_W, 0.0)
        else:
            Q_source_W = max(Q_source_W, 0.0)

        # Temperature after source/sink (uses pipe-loss-adjusted inlet temp)
        if self.C_water > 0:
            T_after_source = T_at_source_inlet + Q_source_W / self.C_water
        else:
            T_after_source = T_at_source_inlet

        # Clamp to physical range
        T_after_source = max(0.0, min(99.0, T_after_source))

        return T_after_source, Q_source_W, Q_pipe1, T_at_source_inlet

    def _update_tank_temperature(self, T_water_out: float, dt: float):
        """
        Update tank temperature based on water returning from HEX.
        Uses a mixing model: tank temp moves toward returning water temp.
        """
        mixing_ratio = self.m_dot_water * dt / self.m_tank
        self.state.T_tank += mixing_ratio * (T_water_out - self.state.T_tank)

    def step(self, mode: SimulationMode) -> SimulationState:
        """Perform one simulation time step"""
        dt = self.config.dt
        self.state.mode = mode

        # Wall heat loss (negative when T_pcm > T_ambient = heat leaves PCM)
        Q_loss = -self._wall_loss_UA * (self.state.T_pcm - self._T_ambient) if self._wall_loss_UA > 0 else 0.0

        # Handle idle mode
        if mode == SimulationMode.IDLE:
            self.state.Q_total = 0
            self.state.Q_fin = 0
            self.state.Q_tube = 0
            self.state.Q_source = 0
            self.state.Q_pipe_loss = 0
            # Apply wall loss even in idle
            if Q_loss != 0.0:
                dH = Q_loss * dt / 1000  # kJ
                self.state.H_pcm_total += dH
                self.state.H_pcm_specific = self.state.H_pcm_total / self.m_pcm
                self.state.T_pcm = self.pcm.enthalpy_data.get_T_from_H(
                    self.state.H_pcm_specific, self.state.curve)
                self._update_phase()
            self.state.Q_loss = Q_loss
            self.state.E_loss += abs(Q_loss) * dt / 1000  # kJ
            self.state.time += dt
            return self.state

        # Determine enthalpy curve based on mode and PCM category
        if mode == SimulationMode.CHARGING:
            if self.pcm.category == PCMCategory.HOT:
                self.state.curve = 'melting'
            else:
                self.state.curve = 'solidifying'
        else:  # DISCHARGING
            if self.pcm.category == PCMCategory.HOT:
                self.state.curve = 'solidifying'
            else:
                self.state.curve = 'melting'

        # Select water inlet temperature based on water supply mode
        Q_pipe_total = 0.0   # Total pipe heat loss this step [W]
        Q_pipe1 = Q_pipe2 = Q_pipe3 = 0.0
        if self.operating.water_supply_mode == WaterSupplyMode.HEAT_SOURCE_SINK:
            T_after_source, Q_source_W, Q_pipe1, T_at_source_inlet = self._calculate_closed_loop_water_in(mode)

            # --- Pipe 2: Source/Sink → HEX Cell (loss to ambient) ---
            T_water_in, Q_pipe2 = self._apply_pipe_loss(T_after_source, self._pipe_UA_2)

            Q_pipe_total = Q_pipe1 + Q_pipe2  # Pipe 3 added after HEX below
        else:
            # Constant temperature mode (existing behavior)
            Q_source_W = 0.0
            if mode == SimulationMode.CHARGING:
                if self.pcm.category == PCMCategory.HOT:
                    T_water_in = self.operating.T_water_hot
                else:
                    T_water_in = self.operating.T_water_cold
            else:  # DISCHARGING
                if self.pcm.category == PCMCategory.HOT:
                    T_water_in = self.operating.T_water_cold
                else:
                    T_water_in = self.operating.T_water_hot

        # Calculate heat transfer
        Q_total, Q_fin, Q_tube, T_water_out = self._calculate_heat_transfer(
            T_water_in,
            self.state.T_pcm,
            self.state.delta_fin,
            self.state.r_front_tube
        )

        # Apply sign convention: positive Q = heat into PCM
        # Use temperature-based sign to handle cases where wall loss
        # pushes T_pcm past T_water (e.g. discharge with wall loss)
        if T_water_in > self.state.T_pcm:
            # Water is hotter → heat flows into PCM
            Q_total = abs(Q_total)
            Q_fin = abs(Q_fin)
            Q_tube = abs(Q_tube)
        else:
            # PCM is hotter → heat flows out of PCM
            Q_total = -abs(Q_total)
            Q_fin = -abs(Q_fin)
            Q_tube = -abs(Q_tube)

        # Update enthalpy (includes wall loss)
        dH = (Q_total + Q_loss) * dt / 1000  # kJ
        self.state.H_pcm_total += dH
        self.state.H_pcm_specific = self.state.H_pcm_total / self.m_pcm

        # Update temperature from enthalpy
        if self.state.is_supercooled:
            # Supercooled: use sensible-only H->T (no latent heat release)
            self.state.T_pcm = self.state.T_supercool_start - \
                (self.state.H_supercool_start - self.state.H_pcm_specific) / self.pcm.cp_sensible

            # Check nucleation trigger
            if self.state.T_pcm <= self.pcm.T_liquidus - self.config.supercooling_deg:
                # Nucleation! Switch back to solidifying curve — temperature jumps up
                self.state.is_supercooled = False
                self.state.T_pcm = self.pcm.enthalpy_data.get_T_from_H(
                    self.state.H_pcm_specific, 'solidifying'
                )
        else:
            self.state.T_pcm = self.pcm.enthalpy_data.get_T_from_H(
                self.state.H_pcm_specific,
                self.state.curve
            )

        # Update front positions
        self._update_front_positions(abs(Q_fin), abs(Q_tube), dt)

        # Update phase
        self._update_phase()

        # Store results
        self.state.Q_total = Q_total
        self.state.Q_fin = Q_fin
        self.state.Q_tube = Q_tube
        self.state.T_water_out = T_water_out
        self.state.E_total += abs(Q_total) * dt / 1000  # kJ
        self.state.T_water_in = T_water_in
        self.state.Q_source = Q_source_W
        self.state.Q_loss = Q_loss
        self.state.E_loss += abs(Q_loss) * dt / 1000  # kJ

        # Update tank temperature in closed-loop mode
        if self.operating.water_supply_mode == WaterSupplyMode.HEAT_SOURCE_SINK:
            # --- Pipe 3: HEX Cell → Tank (loss to ambient) ---
            T_return, Q_pipe3 = self._apply_pipe_loss(T_water_out, self._pipe_UA_3)
            Q_pipe_total += Q_pipe3
            self._update_tank_temperature(T_return, dt)

            # Store loop temperature sensors
            self.state.T_at_source_inlet = T_at_source_inlet
            self.state.T_after_source = T_after_source
            self.state.T_return_to_tank = T_return

            # Store per-segment pipe losses
            self.state.Q_pipe_loss_1 = Q_pipe1
            self.state.Q_pipe_loss_2 = Q_pipe2
            self.state.Q_pipe_loss_3 = Q_pipe3

        # Store pipe loss
        self.state.Q_pipe_loss = Q_pipe_total
        self.state.E_pipe_loss += abs(Q_pipe_total) * dt / 1000  # kJ

        self.state.time += dt

        return self.state

    def run_charging(self, t_max: Optional[float] = None) -> Tuple[SimulationResults, SimulationSummary]:
        """Run charging simulation"""
        if t_max is None:
            t_max = self.config.t_max

        initial_T = self.state.T_pcm

        # Determine target temperature for early termination
        # In heat source/sink mode, skip early termination (run to t_max)
        use_early_termination = (
            self.operating.water_supply_mode == WaterSupplyMode.CONSTANT_TEMPERATURE
        )
        if use_early_termination:
            if self.pcm.category == PCMCategory.HOT:
                target_T = self.operating.T_water_hot
            else:
                target_T = self.operating.T_water_cold

        # Record initial state
        self.results.record(self.state)

        while self.state.time < t_max:
            self.step(SimulationMode.CHARGING)
            self.results.record(self.state)

            # Progress callback
            if self.progress_callback:
                progress = min(self.state.time / t_max, 1.0)
                self.progress_callback(progress, self.state)

            # Check if complete (only in constant temperature mode)
            if use_early_termination:
                if self.pcm.category == PCMCategory.HOT:
                    if (self.state.phase == Phase.FULLY_LIQUID and
                        self.state.T_pcm >= target_T - 0.5):
                        break
                else:
                    if (self.state.phase == Phase.FULLY_SOLID and
                        self.state.T_pcm <= target_T + 0.5):
                        break

        summary = SimulationSummary(
            total_time_s=self.state.time,
            total_time_min=self.state.time / 60,
            final_T_pcm=self.state.T_pcm,
            total_energy_kWh=self.state.E_total / 3600,
            average_power_kW=self.state.E_total / self.state.time if self.state.time > 0 else 0,
            final_melt_fraction=self.state.f_melted_total,
            final_phase=self.state.phase.value,
            initial_T_pcm=initial_T,
            pcm_mass_kg=self.m_pcm,
            mode="Charging"
        )

        return self.results, summary

    def run_discharging(self, t_max: Optional[float] = None) -> Tuple[SimulationResults, SimulationSummary]:
        """Run discharging simulation"""
        if t_max is None:
            t_max = self.config.t_max

        initial_T = self.state.T_pcm

        # Determine target temperature for early termination
        # In heat source/sink mode, skip early termination (run to t_max)
        use_early_termination = (
            self.operating.water_supply_mode == WaterSupplyMode.CONSTANT_TEMPERATURE
        )
        if use_early_termination:
            if self.pcm.category == PCMCategory.HOT:
                target_T = self.operating.T_water_cold
            else:
                target_T = self.operating.T_water_hot

        # Record initial state
        self.results.record(self.state)

        while self.state.time < t_max:
            self.step(SimulationMode.DISCHARGING)
            self.results.record(self.state)

            # Progress callback
            if self.progress_callback:
                progress = min(self.state.time / t_max, 1.0)
                self.progress_callback(progress, self.state)

            # Check if complete (only in constant temperature mode)
            if use_early_termination:
                if self.pcm.category == PCMCategory.HOT:
                    if (self.state.phase == Phase.FULLY_SOLID and
                        self.state.T_pcm <= target_T + 0.5):
                        break
                else:
                    if (self.state.phase == Phase.FULLY_LIQUID and
                        self.state.T_pcm >= target_T - 0.5):
                        break

        summary = SimulationSummary(
            total_time_s=self.state.time,
            total_time_min=self.state.time / 60,
            final_T_pcm=self.state.T_pcm,
            total_energy_kWh=self.state.E_total / 3600,
            average_power_kW=self.state.E_total / self.state.time if self.state.time > 0 else 0,
            final_melt_fraction=self.state.f_melted_total,
            final_phase=self.state.phase.value,
            initial_T_pcm=initial_T,
            pcm_mass_kg=self.m_pcm,
            mode="Discharging"
        )

        return self.results, summary

    def run_full_cycle(self, charge_time: float, discharge_time: float) -> Tuple[SimulationResults, SimulationSummary]:
        """Run a complete charge-discharge cycle"""
        # Charging phase
        self.run_charging(charge_time)

        # Continue with discharging phase
        self.run_discharging(self.state.time + discharge_time)

        summary = SimulationSummary(
            total_time_s=self.state.time,
            total_time_min=self.state.time / 60,
            final_T_pcm=self.state.T_pcm,
            total_energy_kWh=self.state.E_total / 3600,
            average_power_kW=self.state.E_total / self.state.time if self.state.time > 0 else 0,
            final_melt_fraction=self.state.f_melted_total,
            final_phase=self.state.phase.value,
            initial_T_pcm=self.config.T_pcm_initial,
            pcm_mass_kg=self.m_pcm,
            mode="Full Cycle"
        )

        return self.results, summary
