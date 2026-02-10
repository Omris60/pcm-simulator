"""
Geometry Module
===============
Box and Heat Exchanger geometry classes with preset configurations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BoxGeometry:
    """Container box dimensions"""
    length: float       # [mm]
    width: float        # [mm]
    height: float       # [mm]
    name: str = "Custom Box"

    @property
    def volume_m3(self) -> float:
        """Box volume in m3"""
        return (self.length / 1000) * (self.width / 1000) * (self.height / 1000)

    @property
    def volume_liters(self) -> float:
        """Box volume in liters"""
        return self.volume_m3 * 1000

    @property
    def surface_area_m2(self) -> float:
        """Total external surface area in m2 (all 6 faces)"""
        L, W, H = self.length / 1000, self.width / 1000, self.height / 1000
        return 2 * (L * W + L * H + W * H)

    def to_dict(self) -> Dict:
        """Convert to dictionary for display"""
        return {
            "Name": self.name,
            "Length": f"{self.length} mm",
            "Width": f"{self.width} mm",
            "Height": f"{self.height} mm",
            "Volume": f"{self.volume_liters:.1f} L"
        }


@dataclass
class HEXGeometry:
    """Heat exchanger geometry"""
    # Tube parameters
    D_in: float              # Tube inner diameter [mm]
    D_out: float             # Tube outer diameter [mm]
    N_rows: int              # Number of parallel flow paths
    N_levels: int            # Number of tube levels
    L_tube: float            # Tube length (finned section) [mm]
    L_tube_total: float      # Total tube length per circuit [mm]
    tube_pitch: float        # Tube pitch (transverse & longitudinal) [mm]

    # Fin parameters
    t_fin: float             # Fin thickness [mm]
    FPI: float               # Fins per inch
    width_mm: float          # Heat exchanger width [mm]

    # Areas (from geometry calculator)
    A_total: float           # Total heat transfer area [m2]
    A_primary: float         # Primary area (tubes) [m2]
    A_secondary: float       # Secondary area (fins) [m2]
    A_flow_cm2: float        # Water flow area [cm2]

    # Material
    k_fin: float = 205       # Fin thermal conductivity (Aluminum) [W/m*K]
    k_tube: float = 205      # Tube thermal conductivity (Aluminum) [W/m*K]

    name: str = "Custom HEX"

    @property
    def r_out(self) -> float:
        """Tube outer radius [m]"""
        return self.D_out / 2 / 1000

    @property
    def r_in(self) -> float:
        """Tube inner radius [m]"""
        return self.D_in / 2 / 1000

    @property
    def r_max_tube(self) -> float:
        """Maximum radius for tube front (half pitch) [m]"""
        return self.tube_pitch / 2 / 1000

    @property
    def fin_pitch(self) -> float:
        """Fin pitch [mm]"""
        return 25.4 / self.FPI

    @property
    def delta_max_fin(self) -> float:
        """Maximum PCM thickness from fin surface [m]"""
        return (self.fin_pitch - self.t_fin) / 2 / 1000

    @property
    def N_tubes_total(self) -> int:
        """Total number of tubes"""
        return self.N_rows * self.N_levels

    @property
    def N_fins(self) -> int:
        """Approximate number of fins"""
        return int(self.L_tube / 25.4 * self.FPI)

    @property
    def A_flow_m2(self) -> float:
        """Water flow area [m2]"""
        return self.A_flow_cm2 / 10000

    def estimate_hex_volume(self) -> float:
        """
        Estimate HEX volume (tubes + fins) [m3]

        This is the volume displaced by the heat exchanger.
        """
        # --- TUBE VOLUME ---
        total_pipe_length_m = (self.L_tube_total / 1000) * self.N_rows
        V_tubes = np.pi * self.r_out**2 * total_pipe_length_m

        # --- FIN VOLUME ---
        N_fins = int((self.L_tube / 25.4) * self.FPI)

        fin_width = self.width_mm / 1000  # [m]
        fin_height = (self.tube_pitch / 1000) * self.N_levels

        fin_area_gross = fin_width * fin_height

        # Subtract tube holes
        tube_hole_area = np.pi * self.r_out**2 * self.N_rows * self.N_levels
        fin_area_net = fin_area_gross - tube_hole_area

        # Volume of all fins
        V_fins = fin_area_net * (self.t_fin / 1000) * N_fins

        return V_tubes + V_fins

    def to_dict(self) -> Dict:
        """Convert to dictionary for display"""
        return {
            "Name": self.name,
            "Tube ID/OD": f"{self.D_in}/{self.D_out} mm",
            "Rows x Levels": f"{self.N_rows} x {self.N_levels}",
            "Tube Length": f"{self.L_tube} mm",
            "Tube Pitch": f"{self.tube_pitch} mm",
            "Fin Thickness": f"{self.t_fin} mm",
            "FPI": f"{self.FPI}",
            "Total Area": f"{self.A_total:.2f} m2",
            "HEX Volume": f"{self.estimate_hex_volume()*1000:.2f} L"
        }


# =============================================================================
# PRESET GEOMETRIES
# =============================================================================

def get_standard_lordan_hex() -> HEXGeometry:
    """Standard Lordan HEX geometry"""
    return HEXGeometry(
        D_in=8.025,              # mm
        D_out=9.525,             # mm
        N_rows=6,
        N_levels=23,
        L_tube=800,              # mm
        L_tube_total=20027,      # mm
        tube_pitch=25.4,         # mm
        t_fin=0.3,               # mm
        FPI=4,
        width_mm=132.0,
        A_total=20.6035,         # m2
        A_primary=3.5946,        # m2
        A_secondary=17.0090,     # m2
        A_flow_cm2=3.0348,       # cm2
        k_fin=205,
        k_tube=205,
        name="V3 Experiment HEX (Lordan)"
    )


def get_standard_box() -> BoxGeometry:
    """Standard Lordan box geometry"""
    return BoxGeometry(
        length=910,    # mm
        width=152,     # mm
        height=620,    # mm
        name="V3 Experiment Box"
    )


def get_large_lordan_hex() -> HEXGeometry:
    """Large Lordan HEX geometry (26 rows, 18 levels, 574 mm tube)"""
    N_rows = 26
    N_levels = 18
    L_tube = 574.0          # mm
    tube_pitch = 25.4        # mm
    D_in = 8.025             # mm
    D_out = 9.525            # mm
    # Scale areas from standard Lordan HEX per (row×level×L_tube)
    std_units = 6 * 23 * 800          # standard: rows × levels × L_tube
    new_units = N_rows * N_levels * L_tube
    scale = new_units / std_units
    A_primary = 3.5946 * scale         # m2
    A_secondary = 17.0090 * scale      # m2
    A_total = A_primary + A_secondary  # m2
    A_flow_cm2 = 3.0348 * (N_rows / 6)  # scales with rows only
    L_tube_total = N_levels * L_tube + (N_levels - 1) * np.pi * tube_pitch / 2
    return HEXGeometry(
        D_in=D_in,
        D_out=D_out,
        N_rows=N_rows,
        N_levels=N_levels,
        L_tube=L_tube,
        L_tube_total=round(L_tube_total, 1),
        tube_pitch=tube_pitch,
        t_fin=0.3,
        FPI=4,
        width_mm=580.0,
        A_total=round(A_total, 4),
        A_primary=round(A_primary, 4),
        A_secondary=round(A_secondary, 4),
        A_flow_cm2=round(A_flow_cm2, 4),
        k_fin=205,
        k_tube=205,
        name="Commercial HEX (Lordan)"
    )


def get_large_lordan_box() -> BoxGeometry:
    """Large Lordan box geometry"""
    return BoxGeometry(
        length=723,    # mm
        width=602,     # mm
        height=550,    # mm
        name="Commercial Cell (Nir)"
    )



class GeometryPresets:
    """Collection of preset geometries"""

    @staticmethod
    def list_hex_presets() -> list:
        """List available HEX presets"""
        return ["V3 Experiment HEX (Lordan)", "Commercial HEX (Lordan)", "Custom"]

    @staticmethod
    def list_box_presets() -> list:
        """List available box presets"""
        return ["V3 Experiment Box", "Commercial Cell (Nir)", "Custom"]

    @staticmethod
    def get_hex(name: str) -> Optional[HEXGeometry]:
        """Get HEX geometry by name"""
        presets = {
            "V3 Experiment HEX (Lordan)": get_standard_lordan_hex,
            "Commercial HEX (Lordan)": get_large_lordan_hex,
        }
        if name in presets:
            return presets[name]()
        return None

    @staticmethod
    def get_box(name: str) -> Optional[BoxGeometry]:
        """Get box geometry by name"""
        presets = {
            "V3 Experiment Box": get_standard_box,
            "Commercial Cell (Nir)": get_large_lordan_box,
        }
        if name in presets:
            return presets[name]()
        return None
