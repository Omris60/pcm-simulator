"""
PCM Material Library
====================
Database of Phase Change Materials with Hot/Cold categories
and proper charging/discharging logic.
"""

import json
import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from enum import Enum

_CUSTOM_LIBRARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_pcm_library.json")


class PCMCategory(Enum):
    """PCM category based on application"""
    HOT = "Hot"    # Charging = heating/melting (heat storage)
    COLD = "Cold"  # Charging = cooling/solidifying (cold storage)


@dataclass
class EnthalpyData:
    """
    Enthalpy-temperature data for PCM with hysteresis

    Data is DIFFERENTIAL enthalpy (dH per degree C), will be converted to cumulative.
    Outside the defined temperature range, only sensible heat is used (no latent heat).
    """
    temperatures: np.ndarray          # Temperature points [C]
    dH_melting: np.ndarray           # Differential enthalpy for melting [kJ/kg/C]
    dH_solidifying: np.ndarray       # Differential enthalpy for solidifying [kJ/kg/C]
    cp_sensible: float = 2.0         # Sensible heat capacity [kJ/kg*K]

    def __post_init__(self):
        """Build cumulative enthalpy curves"""
        self._build_cumulative_curves()

    def _build_cumulative_curves(self):
        """Convert differential to cumulative enthalpy"""
        n = len(self.temperatures)

        # Cumulative enthalpy (includes sensible heat)
        self.H_melting = np.zeros(n)
        self.H_solidifying = np.zeros(n)

        for i in range(1, n):
            dT = self.temperatures[i] - self.temperatures[i-1]
            # Total enthalpy change = sensible + latent
            self.H_melting[i] = self.H_melting[i-1] + (self.cp_sensible + self.dH_melting[i-1]) * dT
            self.H_solidifying[i] = self.H_solidifying[i-1] + (self.cp_sensible + self.dH_solidifying[i-1]) * dT

    def get_cumulative_H(self, T: float, curve: str = 'melting') -> float:
        """
        Get cumulative enthalpy at temperature T

        Uses linear interpolation within the defined range.
        Extrapolates with sensible heat only outside the range.
        """
        H_curve = self.H_melting if curve == 'melting' else self.H_solidifying

        if T < self.T_min:
            H_at_min = H_curve[0]
            return H_at_min - self.cp_sensible * (self.T_min - T)
        elif T > self.T_max:
            H_at_max = H_curve[-1]
            return H_at_max + self.cp_sensible * (T - self.T_max)
        else:
            return np.interp(T, self.temperatures, H_curve)

    def get_T_from_H(self, H: float, curve: str = 'melting') -> float:
        """
        Get temperature from cumulative enthalpy (inverse lookup)

        Uses linear interpolation within the defined range.
        Extrapolates with sensible heat only outside the range.
        """
        H_curve = self.H_melting if curve == 'melting' else self.H_solidifying
        H_min = H_curve[0]
        H_max = H_curve[-1]

        if H < H_min:
            return self.T_min - (H_min - H) / self.cp_sensible
        elif H > H_max:
            return self.T_max + (H - H_max) / self.cp_sensible
        else:
            return np.interp(H, H_curve, self.temperatures)

    def get_dH_dT(self, T: float, curve: str = 'melting') -> float:
        """
        Get local dH/dT (effective heat capacity) at temperature T

        Returns sensible + latent within the defined range.
        Returns sensible only outside the range.
        """
        if T < self.T_min or T > self.T_max:
            return self.cp_sensible

        dH_curve = self.dH_melting if curve == 'melting' else self.dH_solidifying
        dH_latent = np.interp(T, self.temperatures, dH_curve)
        return self.cp_sensible + dH_latent

    @property
    def T_min(self) -> float:
        return self.temperatures[0]

    @property
    def T_max(self) -> float:
        return self.temperatures[-1]

    @property
    def total_latent_heat_melting(self) -> float:
        """Total latent heat for melting [kJ/kg]"""
        return np.sum(self.dH_melting[:-1] * np.diff(self.temperatures))

    @property
    def total_latent_heat_solidifying(self) -> float:
        """Total latent heat for solidifying [kJ/kg]"""
        return np.sum(self.dH_solidifying[:-1] * np.diff(self.temperatures))


@dataclass
class PCMMaterial:
    """Phase Change Material with all properties"""
    name: str
    category: PCMCategory
    melting_range: Tuple[float, float]  # (T_solidus, T_liquidus)
    rho_solid: float                    # Density solid [kg/m3]
    rho_liquid: float                   # Density liquid [kg/m3]
    k_solid: float                      # Thermal conductivity solid [W/m*K]
    k_liquid: float                     # Thermal conductivity liquid [W/m*K]
    cp_sensible: float                  # Sensible heat capacity [kJ/kg*K]
    enthalpy_data: EnthalpyData
    description: str = ""               # Description/use case

    @property
    def rho_avg(self) -> float:
        """Average density"""
        return (self.rho_solid + self.rho_liquid) / 2

    @property
    def T_solidus(self) -> float:
        return self.melting_range[0]

    @property
    def T_liquidus(self) -> float:
        return self.melting_range[1]

    @property
    def total_latent_heat(self) -> float:
        """Total latent heat [kJ/kg]"""
        return self.enthalpy_data.total_latent_heat_melting

    def get_k_effective(self, f_liquid: float, Nu_convection: float = 1.0) -> float:
        """
        Get effective thermal conductivity based on liquid fraction
        Includes natural convection enhancement in liquid
        """
        k_liquid_eff = self.k_liquid * Nu_convection
        return self.k_solid * (1 - f_liquid) + k_liquid_eff * f_liquid

    def get_charging_curve(self) -> str:
        """
        Get which enthalpy curve to use for charging

        Hot PCM: Charging = heating = melting
        Cold PCM: Charging = cooling = solidifying
        """
        if self.category == PCMCategory.HOT:
            return 'melting'
        else:
            return 'solidifying'

    def get_discharging_curve(self) -> str:
        """
        Get which enthalpy curve to use for discharging

        Hot PCM: Discharging = cooling = solidifying
        Cold PCM: Discharging = heating = melting
        """
        if self.category == PCMCategory.HOT:
            return 'solidifying'
        else:
            return 'melting'


# =============================================================================
# MATERIAL DEFINITIONS
# =============================================================================

def _create_SP50_gel_enthalpy() -> EnthalpyData:
    """Create enthalpy data for SP50_gel from Rubitherm datasheet"""
    temperatures = np.array([40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55], dtype=float)

    # Melting (charging) - from datasheet histogram
    dH_melting_raw = np.array([2, 2, 2, 3, 4, 5, 7, 12, 31, 67, 92, 6, 3, 4, 2, 3], dtype=float)
    scale_melting = 190 / np.sum(dH_melting_raw)
    dH_melting = dH_melting_raw * scale_melting

    # Solidifying (discharging) - from datasheet histogram
    dH_solidifying_raw = np.array([3, 4, 4, 8, 13, 19, 29, 50, 89, 2, 2, 3, 3, 2, 2, 4], dtype=float)
    scale_solidifying = 190 / np.sum(dH_solidifying_raw)
    dH_solidifying = dH_solidifying_raw * scale_solidifying

    return EnthalpyData(
        temperatures=temperatures,
        dH_melting=dH_melting,
        dH_solidifying=dH_solidifying,
        cp_sensible=2.0
    )


def _create_SP50_enthalpy() -> EnthalpyData:
    """
    Create enthalpy data for SP50 (non-gel version)
    Similar to SP50_gel but slightly narrower melting range
    """
    temperatures = np.array([42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54], dtype=float)

    # SP50 has similar properties to gel version, slightly sharper peak
    dH_melting_raw = np.array([2, 3, 5, 8, 15, 35, 70, 95, 15, 5, 3, 2, 2], dtype=float)
    scale_melting = 200 / np.sum(dH_melting_raw)
    dH_melting = dH_melting_raw * scale_melting

    dH_solidifying_raw = np.array([2, 5, 10, 20, 40, 75, 90, 10, 4, 3, 2, 2, 2], dtype=float)
    scale_solidifying = 200 / np.sum(dH_solidifying_raw)
    dH_solidifying = dH_solidifying_raw * scale_solidifying

    return EnthalpyData(
        temperatures=temperatures,
        dH_melting=dH_melting,
        dH_solidifying=dH_solidifying,
        cp_sensible=2.0
    )


def _create_RT5HC_enthalpy() -> EnthalpyData:
    """Create enthalpy data for RT5HC from Rubitherm datasheet"""
    temperatures = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=float)

    dH_melting = np.array([2, 2, 2, 2, 4, 5, 13, 89, 139, 3, 2, 2, 2, 2, 2, 2], dtype=float)
    dH_solidifying = np.array([2, 2, 2, 2, 2, 3, 5, 36, 202, 2, 2, 3, 2, 3, 2, 2], dtype=float)

    return EnthalpyData(
        temperatures=temperatures,
        dH_melting=dH_melting,
        dH_solidifying=dH_solidifying,
        cp_sensible=2.0
    )


def _create_RT10HC_enthalpy() -> EnthalpyData:
    """
    Create enthalpy data for RT10HC (cold storage PCM)
    Melting range: ~8-12C, Latent heat: ~209 kJ/kg
    Based on Rubitherm datasheet enthalpy distribution.
    """
    temperatures = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=float)

    dH_melting = np.array([3, 3, 3, 4, 4, 6, 14, 100, 61, 3, 3, 0, 1, 2, 2], dtype=float)
    dH_solidifying = np.array([3, 2, 2, 1, 1, 2, 4, 40, 145, 2, 2, 3, 2, 2, 2], dtype=float)

    return EnthalpyData(
        temperatures=temperatures,
        dH_melting=dH_melting,
        dH_solidifying=dH_solidifying,
        cp_sensible=2.0
    )


# =============================================================================
# MATERIAL LIBRARY
# =============================================================================

class PCMMaterialLibrary:
    """Library of PCM materials"""

    def __init__(self):
        self._materials: Dict[str, PCMMaterial] = {}
        self._load_default_materials()
        self._load_custom_materials()

    def _load_default_materials(self):
        """Load default PCM materials into library"""

        # SP50_gel - Hot PCM for heat storage
        self._materials["SP50_gel"] = PCMMaterial(
            name="SP50_gel",
            category=PCMCategory.HOT,
            melting_range=(42, 52),
            rho_solid=1500,
            rho_liquid=1400,
            k_solid=0.6,
            k_liquid=0.6,
            cp_sensible=2.0,
            enthalpy_data=_create_SP50_gel_enthalpy(),
            description="Rubitherm SP50 gel, heat storage 42-52C"
        )

        # SP50 - Hot PCM (non-gel version)
        self._materials["SP50"] = PCMMaterial(
            name="SP50",
            category=PCMCategory.HOT,
            melting_range=(46, 52),
            rho_solid=1550,
            rho_liquid=1450,
            k_solid=0.6,
            k_liquid=0.6,
            cp_sensible=2.0,
            enthalpy_data=_create_SP50_enthalpy(),
            description="Rubitherm SP50, heat storage 46-52C"
        )

        # RT5HC - Cold PCM for cold storage (AC applications)
        self._materials["RT5HC"] = PCMMaterial(
            name="RT5HC",
            category=PCMCategory.COLD,
            melting_range=(2, 10),
            rho_solid=880,
            rho_liquid=770,
            k_solid=0.2,
            k_liquid=0.2,
            cp_sensible=2.0,
            enthalpy_data=_create_RT5HC_enthalpy(),
            description="Rubitherm RT5HC, cold storage 2-10C (AC)"
        )

        # RT10HC - Cold PCM for cold storage
        self._materials["RT10HC"] = PCMMaterial(
            name="RT10HC",
            category=PCMCategory.COLD,
            melting_range=(7, 12),
            rho_solid=880,
            rho_liquid=770,
            k_solid=0.2,
            k_liquid=0.2,
            cp_sensible=2.0,
            enthalpy_data=_create_RT10HC_enthalpy(),
            description="Rubitherm RT10HC, cold storage 7-12C"
        )

    def get_material(self, name: str) -> PCMMaterial:
        """Get a PCM material by name"""
        if name not in self._materials:
            raise ValueError(f"Material '{name}' not found. Available: {self.list_materials()}")
        return self._materials[name]

    def list_materials(self) -> List[str]:
        """List all available material names"""
        return list(self._materials.keys())

    def list_hot_materials(self) -> List[str]:
        """List Hot PCM materials"""
        return [name for name, mat in self._materials.items() if mat.category == PCMCategory.HOT]

    def list_cold_materials(self) -> List[str]:
        """List Cold PCM materials"""
        return [name for name, mat in self._materials.items() if mat.category == PCMCategory.COLD]

    def get_materials_by_category(self, category: PCMCategory) -> Dict[str, PCMMaterial]:
        """Get all materials of a given category"""
        return {name: mat for name, mat in self._materials.items() if mat.category == category}

    def add_custom_material(self, material: PCMMaterial) -> None:
        """Add a custom material to the library (in-memory only)"""
        self._materials[material.name] = material

    def save_custom_material(self, material: PCMMaterial) -> None:
        """Save a custom material to the library and persist to disk"""
        self._materials[material.name] = material
        self._save_custom_material_to_disk(material)

    def remove_custom_material(self, name: str) -> None:
        """Remove a custom material from the library and disk"""
        if name in self._materials:
            del self._materials[name]
        custom_data = self._read_custom_file()
        if name in custom_data:
            del custom_data[name]
            self._write_custom_file(custom_data)

    def list_custom_materials(self) -> List[str]:
        """List names of custom (user-saved) materials"""
        custom_data = self._read_custom_file()
        return list(custom_data.keys())

    # --- Persistence helpers ---

    @staticmethod
    def _read_custom_file() -> Dict:
        if not os.path.exists(_CUSTOM_LIBRARY_PATH):
            return {}
        with open(_CUSTOM_LIBRARY_PATH, 'r') as f:
            return json.load(f)

    @staticmethod
    def _write_custom_file(data: Dict) -> None:
        with open(_CUSTOM_LIBRARY_PATH, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_custom_material_to_disk(self, material: PCMMaterial) -> None:
        custom_data = self._read_custom_file()
        custom_data[material.name] = {
            "category": material.category.value,
            "melting_range": list(material.melting_range),
            "rho_solid": material.rho_solid,
            "rho_liquid": material.rho_liquid,
            "k_solid": material.k_solid,
            "k_liquid": material.k_liquid,
            "cp_sensible": material.cp_sensible,
            "description": material.description,
            "enthalpy": {
                "temperatures": material.enthalpy_data.temperatures.tolist(),
                "dH_melting": material.enthalpy_data.dH_melting.tolist(),
                "dH_solidifying": material.enthalpy_data.dH_solidifying.tolist(),
                "cp_sensible": material.enthalpy_data.cp_sensible
            }
        }
        self._write_custom_file(custom_data)

    def _load_custom_materials(self) -> None:
        custom_data = self._read_custom_file()
        for name, props in custom_data.items():
            enthalpy = EnthalpyData(
                temperatures=np.array(props["enthalpy"]["temperatures"], dtype=float),
                dH_melting=np.array(props["enthalpy"]["dH_melting"], dtype=float),
                dH_solidifying=np.array(props["enthalpy"]["dH_solidifying"], dtype=float),
                cp_sensible=props["enthalpy"]["cp_sensible"]
            )
            self._materials[name] = PCMMaterial(
                name=name,
                category=PCMCategory(props["category"]),
                melting_range=tuple(props["melting_range"]),
                rho_solid=props["rho_solid"],
                rho_liquid=props["rho_liquid"],
                k_solid=props["k_solid"],
                k_liquid=props["k_liquid"],
                cp_sensible=props["cp_sensible"],
                enthalpy_data=enthalpy,
                description=props.get("description", "")
            )

    def get_material_info(self, name: str) -> Dict:
        """Get material properties as a dictionary for display"""
        mat = self.get_material(name)
        return {
            "Name": mat.name,
            "Category": mat.category.value,
            "Melting Range": f"{mat.T_solidus}-{mat.T_liquidus} C",
            "Density (solid)": f"{mat.rho_solid} kg/m3",
            "Density (liquid)": f"{mat.rho_liquid} kg/m3",
            "Conductivity (solid)": f"{mat.k_solid} W/m*K",
            "Conductivity (liquid)": f"{mat.k_liquid} W/m*K",
            "Cp (sensible)": f"{mat.cp_sensible} kJ/kg*K",
            "Latent Heat": f"{mat.total_latent_heat:.0f} kJ/kg",
            "Description": mat.description
        }


def get_library() -> PCMMaterialLibrary:
    """Get the PCM material library"""
    return PCMMaterialLibrary()
