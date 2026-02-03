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
        name="Standard Lordan HEX"
    )


def get_standard_box() -> BoxGeometry:
    """Standard Lordan box geometry"""
    return BoxGeometry(
        length=910,    # mm
        width=152,     # mm
        height=620,    # mm
        name="Standard Lordan Box"
    )


def get_compact_hex() -> HEXGeometry:
    """Compact HEX geometry for smaller applications"""
    return HEXGeometry(
        D_in=6.35,               # mm (1/4")
        D_out=7.94,              # mm
        N_rows=4,
        N_levels=15,
        L_tube=500,              # mm
        L_tube_total=8000,       # mm
        tube_pitch=20.0,         # mm
        t_fin=0.25,              # mm
        FPI=5,
        width_mm=100.0,
        A_total=6.5,             # m2
        A_primary=1.2,           # m2
        A_secondary=5.3,         # m2
        A_flow_cm2=1.27,         # cm2
        k_fin=205,
        k_tube=205,
        name="Compact HEX"
    )


def get_compact_box() -> BoxGeometry:
    """Compact box geometry"""
    return BoxGeometry(
        length=600,    # mm
        width=120,     # mm
        height=400,    # mm
        name="Compact Box"
    )


class GeometryPresets:
    """Collection of preset geometries"""

    @staticmethod
    def list_hex_presets() -> list:
        """List available HEX presets"""
        return ["Standard Lordan HEX", "Compact HEX", "Custom"]

    @staticmethod
    def list_box_presets() -> list:
        """List available box presets"""
        return ["Standard Lordan Box", "Compact Box", "Custom"]

    @staticmethod
    def get_hex(name: str) -> Optional[HEXGeometry]:
        """Get HEX geometry by name"""
        presets = {
            "Standard Lordan HEX": get_standard_lordan_hex,
            "Compact HEX": get_compact_hex,
        }
        if name in presets:
            return presets[name]()
        return None

    @staticmethod
    def get_box(name: str) -> Optional[BoxGeometry]:
        """Get box geometry by name"""
        presets = {
            "Standard Lordan Box": get_standard_box,
            "Compact Box": get_compact_box,
        }
        if name in presets:
            return presets[name]()
        return None
