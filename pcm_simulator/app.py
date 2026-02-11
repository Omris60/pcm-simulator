"""
PCM Heat Exchanger Simulator
============================
Streamlit web application for PCM thermal simulation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pcm_library import get_library, PCMCategory, PCMMaterial, EnthalpyData
from geometry import BoxGeometry, HEXGeometry, GeometryPresets
from simulation_core import (
    PCMHeatExchangerSimulation,
    OperatingConditions,
    SimulationConfig,
    WaterProperties,
    SimulationMode,
    WaterSupplyMode,
    SourceControlMode,
    HeatSourceSinkConfig,
    WallLossConfig,
    PipeLossConfig,
    PumpConfig
)
from visualization import (
    create_temperature_plot,
    create_power_plot,
    create_energy_plot,
    create_capacity_plot,
    create_melt_fraction_plot,
    create_front_position_plot,
    create_enthalpy_plot,
    create_combined_dashboard,
    create_enthalpy_curve_plot,
    create_source_power_plot,
    create_wall_loss_plot,
    create_loop_temperature_plot,
    create_pipe_loss_plot,
    create_system_schematic,
    export_data_to_csv,
    create_sweep_heatmap,
    create_sweep_line_plot
)

# Page config
st.set_page_config(
    page_title="PCM Simulator",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .hot-pcm {
        color: #ff4b4b;
        font-weight: bold;
    }
    .cold-pcm {
        color: #4b9bff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'sweep_results' not in st.session_state:
    st.session_state.sweep_results = None


def main():
    st.title("PCM Heat Exchanger Simulator")
    st.markdown("*Transient thermal analysis with moving boundary model*")

    # Get PCM library
    library = get_library()

    # =========================================================================
    # SIDEBAR - INPUT PARAMETERS
    # =========================================================================
    with st.sidebar:
        st.header("Simulation Parameters")

        # ----- PCM Selection -----
        st.subheader("1. PCM Material")

        # Session state for editing a library PCM
        if 'edit_pcm' not in st.session_state:
            st.session_state.edit_pcm = None

        source_options = ["Library", "Custom"]
        default_source_idx = 1 if st.session_state.edit_pcm is not None else 0
        pcm_source = st.radio(
            "PCM Source",
            source_options,
            index=default_source_idx,
            horizontal=True
        )

        # Clear edit state when user switches back to Library
        if pcm_source == "Library" and st.session_state.edit_pcm is not None:
            st.session_state.edit_pcm = None

        if pcm_source == "Library":
            # --- Library PCM selection ---
            category_filter = st.radio(
                "Category",
                ["All", "Hot PCMs", "Cold PCMs"],
                horizontal=True
            )

            if category_filter == "Hot PCMs":
                available_materials = library.list_hot_materials()
            elif category_filter == "Cold PCMs":
                available_materials = library.list_cold_materials()
            else:
                available_materials = library.list_materials()

            selected_pcm_name = st.selectbox(
                "Select Material",
                available_materials,
                index=0
            )

            pcm = library.get_material(selected_pcm_name)

            # Show Edit / Remove buttons for custom materials
            is_custom = selected_pcm_name in library.list_custom_materials()
            if is_custom:
                col_edit, col_remove = st.columns(2)
                with col_edit:
                    if st.button("Edit", use_container_width=True):
                        st.session_state.edit_pcm = selected_pcm_name
                        st.rerun()
                with col_remove:
                    if st.button("Remove", type="secondary", use_container_width=True):
                        st.session_state.confirm_remove = selected_pcm_name

                if getattr(st.session_state, 'confirm_remove', None) == selected_pcm_name:
                    st.warning(f"Delete '{selected_pcm_name}' permanently?")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes, delete", type="primary", use_container_width=True):
                            library.remove_custom_material(selected_pcm_name)
                            st.session_state.confirm_remove = None
                            st.rerun()
                    with col_no:
                        if st.button("Cancel", use_container_width=True):
                            st.session_state.confirm_remove = None
                            st.rerun()

        else:
            # --- Custom PCM creation / editing ---
            # Load defaults from edit_pcm if editing, otherwise use blank defaults
            edit_mat = None
            if st.session_state.edit_pcm is not None:
                try:
                    edit_mat = library.get_material(st.session_state.edit_pcm)
                except ValueError:
                    st.session_state.edit_pcm = None

            defaults = {
                "name": edit_mat.name if edit_mat else "My_Custom_PCM",
                "category": edit_mat.category.value if edit_mat else "Hot",
                "description": edit_mat.description if edit_mat else "Custom PCM material",
                "T_solidus": float(edit_mat.T_solidus) if edit_mat else 45.0,
                "T_liquidus": float(edit_mat.T_liquidus) if edit_mat else 52.0,
                "rho_solid": float(edit_mat.rho_solid) if edit_mat else 1500.0,
                "rho_liquid": float(edit_mat.rho_liquid) if edit_mat else 1400.0,
                "k_solid": float(edit_mat.k_solid) if edit_mat else 0.6,
                "k_liquid": float(edit_mat.k_liquid) if edit_mat else 0.6,
                "cp_sensible": float(edit_mat.cp_sensible) if edit_mat else 2.0,
            }

            if edit_mat:
                st.info(f"Editing: {edit_mat.name}")

            with st.expander("Custom PCM Properties", expanded=True):
                # Section A: Basic info
                custom_name = st.text_input("PCM Name", value=defaults["name"])
                custom_category = st.radio(
                    "Category",
                    ["Hot", "Cold"],
                    index=0 if defaults["category"] == "Hot" else 1,
                    horizontal=True,
                    key="custom_category"
                )
                category_enum = PCMCategory.HOT if custom_category == "Hot" else PCMCategory.COLD
                custom_description = st.text_input("Description", value=defaults["description"])

                st.markdown("---")

                # Section B: Melting range
                st.markdown("**Melting Range**")
                col1, col2 = st.columns(2)
                with col1:
                    T_solidus = st.number_input("T solidus (C)", value=defaults["T_solidus"], min_value=-20.0, max_value=200.0, step=1.0)
                with col2:
                    T_liquidus = st.number_input("T liquidus (C)", value=defaults["T_liquidus"], min_value=-20.0, max_value=200.0, step=1.0)

                if T_liquidus <= T_solidus:
                    st.error("T liquidus must be greater than T solidus!")
                    T_liquidus = T_solidus + 1.0

                st.markdown("---")

                # Section C: Physical properties
                st.markdown("**Physical Properties**")
                col1, col2 = st.columns(2)
                with col1:
                    rho_solid = st.number_input("Density solid (kg/m3)", value=defaults["rho_solid"], min_value=500.0, max_value=3000.0, step=10.0)
                    k_solid = st.number_input("k solid (W/mK)", value=defaults["k_solid"], min_value=0.05, max_value=10.0, step=0.05, format="%.2f")
                with col2:
                    rho_liquid = st.number_input("Density liquid (kg/m3)", value=defaults["rho_liquid"], min_value=500.0, max_value=3000.0, step=10.0)
                    k_liquid = st.number_input("k liquid (W/mK)", value=defaults["k_liquid"], min_value=0.05, max_value=10.0, step=0.05, format="%.2f")
                cp_sensible = st.number_input("Cp sensible (kJ/kgK)", value=defaults["cp_sensible"], min_value=0.5, max_value=10.0, step=0.1, format="%.1f")

                st.markdown("---")

                # Section D: Enthalpy distribution (manual entry)
                st.markdown("**Enthalpy Distribution (kJ/kg/C)**")
                st.caption("Enter dH/dT values for each 1C interval. These are actual latent heat contributions per degree.")

                temperatures = np.arange(T_solidus, T_liquidus + 1, 1.0)
                n_temps = len(temperatures)

                # Build default enthalpy table from edit_mat or zeros
                if edit_mat is not None and T_solidus == defaults["T_solidus"] and T_liquidus == defaults["T_liquidus"]:
                    default_dH_melt = edit_mat.enthalpy_data.dH_melting.copy()
                    default_dH_solid = edit_mat.enthalpy_data.dH_solidifying.copy()
                    # Ensure array lengths match (in case user changed range after load)
                    if len(default_dH_melt) != n_temps:
                        default_dH_melt = np.zeros(n_temps)
                        default_dH_solid = np.zeros(n_temps)
                else:
                    default_dH_melt = np.zeros(n_temps)
                    default_dH_solid = np.zeros(n_temps)

                enthalpy_df = pd.DataFrame({
                    "Temperature (C)": temperatures,
                    "dH Melting (kJ/kg/C)": default_dH_melt,
                    "dH Solidifying (kJ/kg/C)": default_dH_solid
                })
                edited_enthalpy = st.data_editor(
                    enthalpy_df,
                    num_rows="fixed",
                    disabled=["Temperature (C)"],
                    use_container_width=True,
                    key="custom_enthalpy_editor"
                )
                dH_melting = np.clip(np.array(edited_enthalpy["dH Melting (kJ/kg/C)"].values, dtype=float), 0, None)
                dH_solidifying = np.clip(np.array(edited_enthalpy["dH Solidifying (kJ/kg/C)"].values, dtype=float), 0, None)

                # Show totals
                total_melt = float(np.sum(dH_melting))
                total_solid = float(np.sum(dH_solidifying))
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Melting", f"{total_melt:.1f} kJ/kg")
                with col2:
                    st.metric("Total Solidifying", f"{total_solid:.1f} kJ/kg")

                # Live preview chart
                st.markdown("**Enthalpy Distribution Preview**")
                fig_preview = go.Figure()
                fig_preview.add_trace(go.Bar(
                    x=temperatures[:-1] + 0.5,
                    y=dH_melting[:-1],
                    name='Melting',
                    marker_color='red',
                    opacity=0.7,
                    width=0.4,
                    offset=-0.2
                ))
                fig_preview.add_trace(go.Bar(
                    x=temperatures[:-1] + 0.5,
                    y=dH_solidifying[:-1],
                    name='Solidifying',
                    marker_color='blue',
                    opacity=0.7,
                    width=0.4,
                    offset=0.2
                ))
                fig_preview.update_layout(
                    xaxis_title="Temperature (C)",
                    yaxis_title="dH/dT (kJ/kg/C)",
                    barmode='group',
                    height=300,
                    margin=dict(t=30, b=30),
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                )
                st.plotly_chart(fig_preview, use_container_width=True)

            # Build the custom PCM object
            enthalpy = EnthalpyData(
                temperatures=temperatures,
                dH_melting=dH_melting,
                dH_solidifying=dH_solidifying,
                cp_sensible=cp_sensible
            )
            pcm = PCMMaterial(
                name=custom_name,
                category=category_enum,
                melting_range=(T_solidus, T_liquidus),
                rho_solid=rho_solid,
                rho_liquid=rho_liquid,
                k_solid=k_solid,
                k_liquid=k_liquid,
                cp_sensible=cp_sensible,
                enthalpy_data=enthalpy,
                description=custom_description
            )

            # Save to library button
            save_label = "Update in Library" if edit_mat else "Add to Library"
            if st.button(save_label):
                # If editing under a new name, remove the old entry
                if edit_mat and custom_name != edit_mat.name:
                    library.remove_custom_material(edit_mat.name)
                library.save_custom_material(pcm)
                st.session_state.edit_pcm = None
                st.success(f"'{custom_name}' saved to library!")
                st.rerun()

        # Display PCM info
        category_class = "hot-pcm" if pcm.category == PCMCategory.HOT else "cold-pcm"
        st.markdown(f"**Category:** <span class='{category_class}'>{pcm.category.value} PCM</span>",
                   unsafe_allow_html=True)
        st.markdown(f"**Melting Range:** {pcm.T_solidus}-{pcm.T_liquidus} C")
        st.markdown(f"**Latent Heat:** {pcm.total_latent_heat:.0f} kJ/kg")
        if pcm.description:
            st.markdown(f"*{pcm.description}*")

        # Show enthalpy curves in expander
        with st.expander("View Enthalpy Curves"):
            fig_enthalpy = create_enthalpy_curve_plot(pcm)
            st.plotly_chart(fig_enthalpy, use_container_width=True)

        st.divider()

        # ----- Geometry -----
        st.subheader("2. Geometry")

        # Box geometry
        box_preset = st.selectbox(
            "Box Preset",
            GeometryPresets.list_box_presets(),
            index=0
        )

        if box_preset == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                box_length = st.number_input("Length (mm)", value=910, min_value=100)
                box_width = st.number_input("Width (mm)", value=152, min_value=50)
            with col2:
                box_height = st.number_input("Height (mm)", value=620, min_value=100)
            box = BoxGeometry(length=box_length, width=box_width, height=box_height, name="Custom")
        else:
            box = GeometryPresets.get_box(box_preset)
            with st.expander("Box Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Length", f"{box.length:.0f} mm")
                    st.metric("Width", f"{box.width:.0f} mm")
                with col2:
                    st.metric("Height", f"{box.height:.0f} mm")
                    st.metric("Volume", f"{box.volume_liters:.1f} L")

        st.caption(f"Box Volume: {box.volume_liters:.1f} L")

        # HEX geometry
        hex_preset = st.selectbox(
            "HEX Preset",
            GeometryPresets.list_hex_presets(),
            index=0
        )

        if hex_preset == "Custom":
            with st.expander("Custom HEX Parameters"):
                col1, col2 = st.columns(2)
                with col1:
                    hex_D_in = st.number_input("Tube ID (mm)", value=8.025, min_value=1.0)
                    hex_D_out = st.number_input("Tube OD (mm)", value=9.525, min_value=1.0)
                    hex_N_rows = st.number_input("N Rows", value=6, min_value=1)
                    hex_N_levels = st.number_input("N Levels", value=23, min_value=1)
                    hex_L_tube = st.number_input("Tube Length (mm)", value=800, min_value=100)
                with col2:
                    hex_tube_pitch = st.number_input("Tube Pitch (mm)", value=25.4, min_value=5.0)
                    hex_t_fin = st.number_input("Fin Thickness (mm)", value=0.3, min_value=0.1)
                    hex_FPI = st.number_input("FPI", value=4.0, min_value=1.0)
                    hex_width = st.number_input("HEX Width (mm)", value=132.0, min_value=50.0)

                # Straight sections + semicircular U-bends between levels
                hex_L_tube_total_default = hex_N_levels * hex_L_tube + (hex_N_levels - 1) * np.pi * hex_tube_pitch / 2
                hex_L_tube_total = st.number_input(
                    "Total tube length per circuit (mm)",
                    value=float(round(hex_L_tube_total_default)),
                    min_value=float(hex_L_tube),
                    help="Total tube length for one flow path including U-bends"
                )
                # Approximate areas
                hex_A_primary = np.pi * (hex_D_out/1000) * (hex_L_tube/1000) * hex_N_rows * hex_N_levels
                hex_A_secondary = hex_A_primary * 4.7  # Typical ratio
                hex_A_total = hex_A_primary + hex_A_secondary
                hex_A_flow = np.pi * (hex_D_in/1000/2)**2 * hex_N_rows * 10000

                hex_geom = HEXGeometry(
                    D_in=hex_D_in, D_out=hex_D_out, N_rows=hex_N_rows, N_levels=hex_N_levels,
                    L_tube=hex_L_tube, L_tube_total=hex_L_tube_total, tube_pitch=hex_tube_pitch,
                    t_fin=hex_t_fin, FPI=hex_FPI, width_mm=hex_width,
                    A_total=hex_A_total, A_primary=hex_A_primary, A_secondary=hex_A_secondary,
                    A_flow_cm2=hex_A_flow, name="Custom HEX"
                )
        else:
            hex_geom = GeometryPresets.get_hex(hex_preset)
            with st.expander("HEX Parameters"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tube ID", f"{hex_geom.D_in} mm")
                    st.metric("Tube OD", f"{hex_geom.D_out} mm")
                    st.metric("Tube Pitch", f"{hex_geom.tube_pitch} mm")
                with col2:
                    st.metric("N Rows", f"{hex_geom.N_rows}")
                    st.metric("N Levels", f"{hex_geom.N_levels}")
                    st.metric("Tube Length", f"{hex_geom.L_tube:.0f} mm")
                with col3:
                    st.metric("L Total", f"{hex_geom.L_tube_total:.0f} mm")
                    st.metric("Width", f"{hex_geom.width_mm:.0f} mm")
                    st.metric("FPI", f"{hex_geom.FPI}")
                st.caption(
                    f"A_total: {hex_geom.A_total:.2f} m\u00b2 | "
                    f"A_primary: {hex_geom.A_primary:.2f} m\u00b2 | "
                    f"A_secondary: {hex_geom.A_secondary:.2f} m\u00b2 | "
                    f"A_flow: {hex_geom.A_flow_cm2:.2f} cm\u00b2"
                )

        st.caption(f"HEX Area: {hex_geom.A_total:.1f} m2")

        # Number of cells in parallel
        n_cells = st.number_input("Number of Cells", value=1, min_value=1, max_value=100, step=1)

        st.divider()

        # ----- Operating Conditions -----
        st.subheader("3. Operating Conditions")

        Q_water = st.number_input("Water Flow Rate (lpm)", value=20.0, min_value=1.0, max_value=500.0)

        # Water supply mode
        water_supply_mode_str = st.radio(
            "Water Supply Mode",
            ["Constant Temperature", "Heat Source/Sink"],
            index=1,
            horizontal=True
        )
        water_supply_mode = (WaterSupplyMode.HEAT_SOURCE_SINK
                             if water_supply_mode_str == "Heat Source/Sink"
                             else WaterSupplyMode.CONSTANT_TEMPERATURE)

        # Set default temperatures based on PCM category
        if pcm.category == PCMCategory.HOT:
            default_T_hot = pcm.T_liquidus + 3
            default_T_cold = pcm.T_solidus - 4
            default_T_init = pcm.T_solidus - 4  # Start cold for charging
        else:  # COLD
            default_T_hot = pcm.T_liquidus + 5
            default_T_cold = pcm.T_solidus - 5
            default_T_init = pcm.T_liquidus + 3  # Start warm for charging

        heat_source_config = None

        if water_supply_mode == WaterSupplyMode.CONSTANT_TEMPERATURE:
            col1, col2 = st.columns(2)
            with col1:
                T_hot = st.number_input("Hot Water Temp (C)", value=float(default_T_hot), min_value=-10.0, max_value=100.0)
            with col2:
                T_cold = st.number_input("Cold Water Temp (C)", value=float(default_T_cold), min_value=-10.0, max_value=100.0)
        else:
            # Heat Source/Sink mode inputs
            source_power = st.number_input("Source/Sink Power (kW)", value=5.0, min_value=0.1, max_value=100.0, step=0.5)
            tank_volume = st.number_input("Tank Volume (L)", value=100.0, min_value=1.0, max_value=1000.0, step=5.0)
            T_tank_initial = st.number_input("Initial Water Temp (C)", value=float(default_T_init), min_value=0.0, max_value=99.0)

            source_control_str = st.radio(
                "Control Mode",
                ["Constant Power", "Thermostat"],
                index=1,
                horizontal=True
            )
            source_control_mode = (SourceControlMode.THERMOSTAT
                                   if source_control_str == "Thermostat"
                                   else SourceControlMode.CONSTANT_POWER)

            T_setpoint = 60.0
            if source_control_mode == SourceControlMode.THERMOSTAT:
                T_setpoint = st.number_input("Setpoint Temp (C)", value=60.0, min_value=0.0, max_value=99.0)

            heat_source_config = HeatSourceSinkConfig(
                power_kW=source_power,
                tank_volume_L=tank_volume,
                T_tank_initial=T_tank_initial,
                control_mode=source_control_mode,
                T_setpoint=T_setpoint
            )
            # Set dummy T_hot/T_cold (unused in this mode)
            T_hot = 0.0
            T_cold = 0.0

        st.divider()

        # ----- Wall Heat Loss -----
        st.subheader("3b. Tank Wall Heat Loss")
        wall_loss_enabled = st.checkbox("Enable Wall Heat Loss", value=False)

        wall_loss_config = None
        if wall_loss_enabled:
            T_ambient = st.number_input("Ambient Temperature (C)", value=25.0, min_value=-10.0, max_value=50.0, step=1.0)

            k_wall = st.number_input("Wall Conductivity (W/m·K)", value=0.3, min_value=0.01, max_value=500.0, step=0.1)

            wall_thickness = st.number_input("Wall Thickness (mm)", value=20.0, min_value=0.5, max_value=50.0, step=0.5)

            wall_loss_config = WallLossConfig(
                enabled=True,
                T_ambient=T_ambient,
                wall_thickness_mm=wall_thickness,
                k_wall=k_wall,
                h_ext=10.0
            )

            # Display computed UA and estimated loss
            R_wall = (wall_thickness / 1000) / k_wall + 1.0 / 10.0
            UA = box.surface_area_m2 / R_wall
            est_loss = UA * abs(default_T_init - T_ambient)
            st.caption(
                f"Surface area: {box.surface_area_m2:.2f} m² | "
                f"UA: {UA:.1f} W/K | "
                f"Est. initial loss: ~{est_loss:.0f} W"
            )

        # ----- Pipe Heat Loss (closed-loop mode only) -----
        pipe_loss_config = None
        if water_supply_mode == WaterSupplyMode.HEAT_SOURCE_SINK:
            st.divider()
            st.subheader("3c. Pipe Heat Loss")
            pipe_loss_enabled = st.checkbox("Enable Pipe Heat Loss", value=False)

            if pipe_loss_enabled:
                pipe_T_ambient = st.number_input(
                    "Pipe Ambient Temperature (°C)",
                    value=25.0, min_value=-10.0, max_value=50.0, step=1.0,
                    key="pipe_T_ambient"
                )

                pipe_OD = st.number_input(
                    "Pipe OD (mm)", value=30.0, min_value=5.0, max_value=200.0, step=1.0
                )

                insulation_presets = {
                    "Fiberglass": 0.04,
                    "Closed-cell foam": 0.025,
                    "Mineral wool": 0.038,
                    "Rubber": 0.032,
                    "None (bare pipe)": 0.0,
                    "Custom": None,
                }
                insulation_material = st.selectbox(
                    "Insulation Material",
                    list(insulation_presets.keys()),
                    index=0
                )

                if insulation_material == "Custom":
                    k_ins = st.number_input(
                        "Insulation Conductivity (W/m·K)",
                        value=0.04, min_value=0.01, max_value=1.0, step=0.005, format="%.3f"
                    )
                elif insulation_material == "None (bare pipe)":
                    k_ins = 0.04  # placeholder; thickness will be 0
                else:
                    k_ins = insulation_presets[insulation_material]

                if insulation_material == "None (bare pipe)":
                    ins_thickness = 0.0
                else:
                    ins_thickness = st.number_input(
                        "Insulation Thickness (mm)",
                        value=13.0, min_value=0.0, max_value=100.0, step=1.0
                    )

                st.markdown("**Pipe Segment Lengths**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    L1 = st.number_input("Tank → Source (m)", value=20.0, min_value=0.0, max_value=500.0, step=1.0)
                with col2:
                    L2 = st.number_input("Source → Cell (m)", value=20.0, min_value=0.0, max_value=500.0, step=1.0)
                with col3:
                    L3 = st.number_input("Cell → Tank (m)", value=5.0, min_value=0.0, max_value=500.0, step=1.0)

                pipe_loss_config = PipeLossConfig(
                    enabled=True,
                    T_ambient=pipe_T_ambient,
                    pipe_OD_mm=pipe_OD,
                    insulation_thickness_mm=ins_thickness,
                    k_insulation=k_ins,
                    h_ext=10.0,
                    L_tank_to_source_m=L1,
                    L_source_to_hex_m=L2,
                    L_hex_to_tank_m=L3,
                )

                # Display estimated pipe UA
                r_pipe = (pipe_OD / 2) / 1000
                r_outer = r_pipe + ins_thickness / 1000
                total_UA = 0.0
                for L in [L1, L2, L3]:
                    if L <= 0:
                        continue
                    R_ins = 0.0
                    if ins_thickness > 0 and k_ins > 0:
                        R_ins = np.log(r_outer / r_pipe) / (2 * np.pi * k_ins * L)
                    R_ext = 1.0 / (2 * np.pi * r_outer * L * 10.0)
                    total_UA += 1.0 / (R_ins + R_ext)

                C_water_est = (Q_water / 60 * 1000 / 1000) * 4180  # ṁ·cp
                if C_water_est > 0:
                    est_dT = total_UA * abs(T_tank_initial - pipe_T_ambient) / C_water_est
                else:
                    est_dT = 0
                st.caption(
                    f"Total pipe UA: {total_UA:.1f} W/K | "
                    f"Est. total ΔT: ~{est_dT:.1f} °C"
                )

        # ----- Circulation Pump (closed-loop mode only) -----
        pump_config = None
        if water_supply_mode == WaterSupplyMode.HEAT_SOURCE_SINK:
            st.divider()
            st.subheader("3d. Circulation Pump")
            pump_enabled = st.checkbox("Enable Pump Heat", value=False)

            if pump_enabled:
                pump_power_W = st.number_input(
                    "Pump Power (W)", value=100.0, min_value=1.0, max_value=5000.0, step=10.0
                )
                pump_config = PumpConfig(enabled=True, power_W=pump_power_W)

                # Show estimated dT
                C_water_est = (Q_water / 60 * 1000 / 1000) * 4180  # ṁ·cp
                if C_water_est > 0:
                    est_pump_dT = pump_power_W / C_water_est
                else:
                    est_pump_dT = 0
                st.caption(f"Est. pump ΔT: ~{est_pump_dT:.2f} °C")

        st.divider()

        # ----- Simulation Settings -----
        st.subheader("4. Simulation Settings")

        sim_mode = st.selectbox(
            "Mode",
            ["Charging", "Discharging", "Full Cycle"]
        )

        T_pcm_init = st.number_input(
            "Initial PCM Temp (C)",
            value=float(default_T_init),
            min_value=-10.0,
            max_value=100.0
        )

        col1, col2 = st.columns(2)
        with col1:
            time_unit = st.selectbox("Time Unit", ["Minutes", "Hours", "Days"], index=0)
        with col2:
            if time_unit == "Minutes":
                t_max_value = st.number_input("Max Time", value=120, min_value=1, step=10, key="t_max")
            elif time_unit == "Hours":
                t_max_value = st.number_input("Max Time", value=2.0, min_value=0.1, step=0.5, key="t_max")
            else:  # Days
                t_max_value = st.number_input("Max Time", value=1.0, min_value=0.1, step=0.5, key="t_max")

        if time_unit == "Minutes":
            t_max_seconds = t_max_value * 60
        elif time_unit == "Hours":
            t_max_seconds = t_max_value * 3600
        else:
            t_max_seconds = t_max_value * 86400

        dt = st.number_input("Time Step (s)", value=1.0, min_value=0.1, max_value=600.0, step=1.0)

        supercooling_deg = st.number_input(
            "Supercooling (°C)",
            value=0.0,
            min_value=0.0,
            max_value=20.0,
            step=0.5,
            help="Degrees below T_liquidus the PCM can drop before nucleation triggers solidification. 0 = no supercooling."
        )

        st.divider()

        # ----- Run Button -----
        run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)

    # =========================================================================
    # MAIN AREA - RESULTS
    # =========================================================================

    # Create operating conditions and config
    operating = OperatingConditions(
        Q_water_lpm=Q_water,
        T_water_hot=T_hot,
        T_water_cold=T_cold,
        water_supply_mode=water_supply_mode,
        heat_source_config=heat_source_config
    )

    config = SimulationConfig(
        dt=dt,
        t_max=t_max_seconds,
        T_pcm_initial=T_pcm_init,
        supercooling_deg=supercooling_deg,
        n_cells=n_cells,
        wall_loss=wall_loss_config,
        pipe_loss=pipe_loss_config,
        pump=pump_config
    )

    # System info columns
    n_info_cols = 5 if n_cells > 1 else 4
    info_cols = st.columns(n_info_cols)

    # Calculate system parameters for display
    hex_volume = hex_geom.estimate_hex_volume()
    V_pcm = box.volume_m3 - hex_volume
    m_pcm = V_pcm * pcm.rho_liquid
    E_latent = m_pcm * pcm.total_latent_heat / 3600  # kWh

    # Calculate pressure drop
    water = WaterProperties()
    m_dot_water = Q_water / 60 * water.rho / 1000  # kg/s
    D_in_m = hex_geom.D_in / 1000
    A_flow_m2 = hex_geom.A_flow_m2
    V_water = m_dot_water / (water.rho * A_flow_m2)
    Re_water = water.rho * V_water * D_in_m / water.mu
    L_tube_total_m = hex_geom.L_tube_total / 1000

    # Friction factor (Blasius for turbulent, Hagen-Poiseuille for laminar)
    if Re_water < 2300:
        f = 64 / Re_water if Re_water > 0 else 0
    else:
        f = 0.316 * (Re_water ** (-0.25))

    # Pressure drop [kPa]
    delta_P_kPa = f * (L_tube_total_m / D_in_m) * water.rho * (V_water ** 2) / 2 / 1000

    with info_cols[0]:
        st.metric("PCM Mass", f"{m_pcm:.1f} kg" if n_cells == 1 else f"{m_pcm:.1f} kg/cell")
    with info_cols[1]:
        st.metric("PCM Volume", f"{V_pcm*1000:.1f} L" if n_cells == 1 else f"{V_pcm*1000:.1f} L/cell")
    with info_cols[2]:
        st.metric("Latent Capacity", f"{E_latent:.2f} kWh" if n_cells == 1 else f"{E_latent:.2f} kWh/cell")
    with info_cols[3]:
        st.metric("Pressure Drop", f"{delta_P_kPa:.1f} kPa")
    if n_cells > 1:
        with info_cols[4]:
            st.metric("Array Total", f"{n_cells} cells / {E_latent * n_cells:.2f} kWh")

    # System schematic
    st.divider()
    st.subheader("System Schematic")
    fig_schematic = create_system_schematic(pcm, box, hex_geom, operating, m_pcm, E_latent,
                                            pipe_loss_config=pipe_loss_config,
                                            pump_config=pump_config)
    st.plotly_chart(fig_schematic, use_container_width=True)

    # Run simulation
    if run_clicked:
        with st.spinner("Running simulation..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(progress, state):
                progress_bar.progress(progress)
                status_text.text(f"Time: {state.time/60:.1f} min | T_PCM: {state.T_pcm:.1f}C | "
                                f"Melt: {state.f_melted_total*100:.0f}%")

            # Create simulation
            sim = PCMHeatExchangerSimulation(
                hex_geometry=hex_geom,
                box_geometry=box,
                pcm=pcm,
                operating=operating,
                config=config,
                progress_callback=progress_callback
            )

            # Run based on mode
            if sim_mode == "Charging":
                results, summary = sim.run_charging()
            elif sim_mode == "Discharging":
                results, summary = sim.run_discharging()
            else:  # Full Cycle
                results, summary = sim.run_full_cycle(t_max_seconds / 2, t_max_seconds / 2)  # Half time each

            st.session_state.results = results
            st.session_state.summary = summary
            st.session_state.simulation_run = True

            progress_bar.progress(1.0)
            status_text.text("Simulation complete!")

    # Display results
    if st.session_state.simulation_run and st.session_state.results is not None:
        results = st.session_state.results
        summary = st.session_state.summary
        data = results.to_numpy()
        if n_cells > 1:
            data['n_cells'] = n_cells
            data['Q_array_kW'] = data['Q_total_kW'] * n_cells
            data['E_array_kWh'] = data['E_total_kWh'] * n_cells
            data['capacity_array_kWh'] = data['capacity_kWh'] * n_cells
        is_source_sink = water_supply_mode == WaterSupplyMode.HEAT_SOURCE_SINK

        st.divider()

        # Summary metrics
        st.subheader("Results Summary")
        col1, col2, col3, col4 = st.columns(4)

        energy_label = "Energy (cell)" if n_cells > 1 else "Energy"
        power_label = "Avg Power (cell)" if n_cells > 1 else "Avg Power"

        with col1:
            st.metric("Total Time", f"{summary.total_time_min:.1f} min")
        with col2:
            st.metric(energy_label, f"{summary.total_energy_kWh:.2f} kWh")
        with col3:
            st.metric(power_label, f"{summary.average_power_kW:.2f} kW")
        with col4:
            st.metric("Final T_PCM", f"{summary.final_T_pcm:.1f} C")

        if n_cells > 1:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Energy (array)", f"{summary.total_energy_kWh * n_cells:.2f} kWh")
            with col2:
                st.metric("Avg Power (array)", f"{summary.average_power_kW * n_cells:.2f} kW")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Phase", summary.final_phase)
        with col2:
            st.metric("Melt Fraction", f"{summary.final_melt_fraction*100:.0f}%")

        initial_cap = data['capacity_kWh'][0] if len(data['capacity_kWh']) > 0 else 0
        final_cap = data['capacity_kWh'][-1] if len(data['capacity_kWh']) > 0 else 0
        cap_label_init = "Initial Capacity (cell)" if n_cells > 1 else "Initial Capacity"
        cap_label_final = "Final Capacity (cell)" if n_cells > 1 else "Final Capacity"
        col1, col2 = st.columns(2)
        with col1:
            st.metric(cap_label_init, f"{initial_cap:.3f} kWh")
        with col2:
            st.metric(cap_label_final, f"{final_cap:.3f} kWh")

        if is_source_sink:
            final_T_tank = data['T_tank_C'][-1] if len(data['T_tank_C']) > 0 else 0
            st.metric("Final Tank Temperature", f"{final_T_tank:.1f} C")

        if wall_loss_config and wall_loss_config.enabled:
            final_E_loss = data['E_loss_kWh'][-1] if len(data['E_loss_kWh']) > 0 else 0
            st.metric("Cumulative Wall Loss", f"{final_E_loss:.3f} kWh")

        if pipe_loss_config and pipe_loss_config.enabled:
            final_E_pipe = data['E_pipe_loss_kWh'][-1] if len(data['E_pipe_loss_kWh']) > 0 else 0
            st.metric("Cumulative Pipe Loss", f"{final_E_pipe:.3f} kWh")

        if pump_config and pump_config.enabled:
            avg_Q_pump = np.mean(data['Q_pump_kW']) if len(data['Q_pump_kW']) > 0 else 0
            st.metric("Pump Heat Input", f"{avg_Q_pump:.3f} kW (avg)")

        st.divider()

        # Plots
        st.subheader("Simulation Plots")

        # Plot selection
        plot_type = st.radio(
            "View",
            ["Dashboard", "Individual Plots"],
            horizontal=True
        )

        # Determine plot parameters based on water supply mode
        plot_T_hot = None if is_source_sink else T_hot
        plot_T_cold = None if is_source_sink else T_cold

        if plot_type == "Dashboard":
            fig = create_combined_dashboard(
                data,
                T_water_hot=plot_T_hot,
                T_water_cold=plot_T_cold,
                delta_max_mm=hex_geom.delta_max_fin * 1000,
                title=f"PCM Simulation: {pcm.name} - {sim_mode}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Individual plots in tabs
            tab_names = ["Temperature", "Power", "Energy", "Capacity", "Melt Fraction", "Front Position", "Enthalpy"]
            is_wall_loss = wall_loss_config and wall_loss_config.enabled
            is_pipe_loss = pipe_loss_config and pipe_loss_config.enabled
            if is_wall_loss:
                tab_names.append("Wall Loss")
            if is_source_sink:
                tab_names.append("Source Power")
                tab_names.append("Loop Temperatures")
            if is_source_sink and is_pipe_loss:
                tab_names.append("Pipe Loss")

            tabs = st.tabs(tab_names)

            with tabs[0]:
                fig = create_temperature_plot(data, plot_T_hot, plot_T_cold)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                fig = create_power_plot(data)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[2]:
                fig = create_energy_plot(data)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[3]:
                fig = create_capacity_plot(data)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[4]:
                fig = create_melt_fraction_plot(data)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[5]:
                fig = create_front_position_plot(data, hex_geom.delta_max_fin * 1000)
                st.plotly_chart(fig, use_container_width=True)

            with tabs[6]:
                fig = create_enthalpy_plot(data)
                st.plotly_chart(fig, use_container_width=True)

            next_tab = 7
            if is_wall_loss:
                with tabs[next_tab]:
                    fig = create_wall_loss_plot(data)
                    st.plotly_chart(fig, use_container_width=True)
                next_tab += 1

            if is_source_sink:
                with tabs[next_tab]:
                    fig = create_source_power_plot(data)
                    st.plotly_chart(fig, use_container_width=True)
                next_tab += 1

                with tabs[next_tab]:
                    fig = create_loop_temperature_plot(data)
                    st.plotly_chart(fig, use_container_width=True)
                next_tab += 1

            if is_source_sink and is_pipe_loss:
                with tabs[next_tab]:
                    fig = create_pipe_loss_plot(data)
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Export options
        st.subheader("Export Data")
        col1, col2 = st.columns(2)

        with col1:
            csv_data = export_data_to_csv(data)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"pcm_simulation_{pcm.name}_{sim_mode.lower()}.csv",
                mime="text/csv"
            )

        with col2:
            # Show data preview
            with st.expander("Preview Data"):
                df = results.to_dataframe()
                st.dataframe(df.head(20))

        # =================================================================
        # PARAMETRIC SWEEP
        # =================================================================
        st.divider()
        st.subheader("Parametric Sweep")

        with st.expander("Sweep Configuration"):
            st.markdown("**Flow Rate Range**")
            sw_col1, sw_col2, sw_col3 = st.columns(3)
            with sw_col1:
                sweep_flow_min = st.number_input("Min Flow (lpm)", value=1.0, min_value=0.1, max_value=500.0, step=1.0, key="sw_flow_min")
            with sw_col2:
                sweep_flow_max = st.number_input("Max Flow (lpm)", value=50.0, min_value=0.1, max_value=500.0, step=1.0, key="sw_flow_max")
            with sw_col3:
                sweep_flow_n = st.number_input("# Steps", value=10, min_value=3, max_value=30, step=1, key="sw_flow_n")

            if is_source_sink:
                st.markdown("**Source Power Range**")
                sp_col1, sp_col2, sp_col3 = st.columns(3)
                with sp_col1:
                    sweep_power_min = st.number_input("Min Power (kW)", value=1.0, min_value=0.1, max_value=100.0, step=0.5, key="sw_pow_min")
                with sp_col2:
                    sweep_power_max = st.number_input("Max Power (kW)", value=20.0, min_value=0.1, max_value=100.0, step=0.5, key="sw_pow_max")
                with sp_col3:
                    sweep_power_n = st.number_input("# Steps", value=10, min_value=3, max_value=30, step=1, key="sw_pow_n")

            run_sweep = st.button("Run Sweep", type="secondary", use_container_width=True)

        if run_sweep:
            flow_rates = np.linspace(sweep_flow_min, sweep_flow_max, int(sweep_flow_n))

            if is_source_sink:
                # 2D sweep: flow rate x source power
                source_powers = np.linspace(sweep_power_min, sweep_power_max, int(sweep_power_n))
                total_runs = len(flow_rates) * len(source_powers)

                avg_power_mat = np.zeros((len(flow_rates), len(source_powers)))
                energy_mat = np.zeros_like(avg_power_mat)
                time_mat = np.zeros_like(avg_power_mat)
                melt_mat = np.zeros_like(avg_power_mat)

                progress = st.progress(0)
                run_count = 0

                for i, flow in enumerate(flow_rates):
                    for j, power in enumerate(source_powers):
                        sweep_hs_config = HeatSourceSinkConfig(
                            power_kW=power,
                            tank_volume_L=operating.heat_source_config.tank_volume_L,
                            T_tank_initial=operating.heat_source_config.T_tank_initial,
                            control_mode=operating.heat_source_config.control_mode,
                            T_setpoint=operating.heat_source_config.T_setpoint,
                        )
                        sweep_op = OperatingConditions(
                            Q_water_lpm=flow,
                            T_water_hot=operating.T_water_hot,
                            T_water_cold=operating.T_water_cold,
                            water_supply_mode=operating.water_supply_mode,
                            heat_source_config=sweep_hs_config,
                        )
                        sweep_sim = PCMHeatExchangerSimulation(
                            hex_geometry=hex_geom,
                            box_geometry=box,
                            pcm=pcm,
                            operating=sweep_op,
                            config=config,
                        )
                        if sim_mode == "Charging":
                            _, s = sweep_sim.run_charging()
                        elif sim_mode == "Discharging":
                            _, s = sweep_sim.run_discharging()
                        else:
                            _, s = sweep_sim.run_full_cycle(config.t_max / 2, config.t_max / 2)

                        avg_power_mat[i, j] = s.average_power_kW
                        energy_mat[i, j] = s.total_energy_kWh
                        time_mat[i, j] = s.total_time_min
                        melt_mat[i, j] = s.final_melt_fraction * 100

                        run_count += 1
                        progress.progress(run_count / total_runs)

                st.session_state.sweep_results = {
                    'mode': '2D',
                    'flow_rates': flow_rates,
                    'source_powers': source_powers,
                    'avg_power': avg_power_mat,
                    'energy': energy_mat,
                    'time': time_mat,
                    'melt': melt_mat,
                }
            else:
                # 1D sweep: flow rate only (constant temperature mode)
                total_runs = len(flow_rates)
                avg_power_arr = np.zeros(total_runs)
                energy_arr = np.zeros(total_runs)
                time_arr = np.zeros(total_runs)
                melt_arr = np.zeros(total_runs)

                progress = st.progress(0)

                for i, flow in enumerate(flow_rates):
                    sweep_op = OperatingConditions(
                        Q_water_lpm=flow,
                        T_water_hot=operating.T_water_hot,
                        T_water_cold=operating.T_water_cold,
                        water_supply_mode=operating.water_supply_mode,
                        heat_source_config=None,
                    )
                    sweep_sim = PCMHeatExchangerSimulation(
                        hex_geometry=hex_geom,
                        box_geometry=box,
                        pcm=pcm,
                        operating=sweep_op,
                        config=config,
                    )
                    if sim_mode == "Charging":
                        _, s = sweep_sim.run_charging()
                    elif sim_mode == "Discharging":
                        _, s = sweep_sim.run_discharging()
                    else:
                        _, s = sweep_sim.run_full_cycle(config.t_max / 2, config.t_max / 2)

                    avg_power_arr[i] = s.average_power_kW
                    energy_arr[i] = s.total_energy_kWh
                    time_arr[i] = s.total_time_min
                    melt_arr[i] = s.final_melt_fraction * 100

                    progress.progress((i + 1) / total_runs)

                st.session_state.sweep_results = {
                    'mode': '1D',
                    'flow_rates': flow_rates,
                    'avg_power': avg_power_arr,
                    'energy': energy_arr,
                    'time': time_arr,
                    'melt': melt_arr,
                }

        # Display sweep results
        if st.session_state.sweep_results is not None:
            sr = st.session_state.sweep_results
            if sr['mode'] == '2D':
                tab_names_sw = ["Avg Power", "Energy", "Time", "Melt Fraction"]
                tabs_sw = st.tabs(tab_names_sw)
                with tabs_sw[0]:
                    fig = create_sweep_heatmap(sr['flow_rates'], sr['source_powers'],
                                               sr['avg_power'], "Avg Power", "kW",
                                               "Average Cell Power vs Flow Rate & Source Power")
                    st.plotly_chart(fig, use_container_width=True)
                with tabs_sw[1]:
                    fig = create_sweep_heatmap(sr['flow_rates'], sr['source_powers'],
                                               sr['energy'], "Energy", "kWh",
                                               "Total Energy vs Flow Rate & Source Power")
                    st.plotly_chart(fig, use_container_width=True)
                with tabs_sw[2]:
                    fig = create_sweep_heatmap(sr['flow_rates'], sr['source_powers'],
                                               sr['time'], "Time", "min",
                                               "Simulation Time vs Flow Rate & Source Power")
                    st.plotly_chart(fig, use_container_width=True)
                with tabs_sw[3]:
                    fig = create_sweep_heatmap(sr['flow_rates'], sr['source_powers'],
                                               sr['melt'], "Melt Fraction", "%",
                                               "Final Melt Fraction vs Flow Rate & Source Power")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # 1D line plot
                metrics = {
                    "Avg Power": (sr['avg_power'], "kW"),
                    "Energy": (sr['energy'], "kWh"),
                    "Time": (sr['time'], "min"),
                    "Melt Fraction": (sr['melt'], "%"),
                }
                fig = create_sweep_line_plot(sr['flow_rates'], metrics,
                                             "Flow Rate (lpm)",
                                             "Parametric Sweep: Flow Rate")
                st.plotly_chart(fig, use_container_width=True)

    else:
        # Show instructions
        st.info("Configure simulation parameters in the sidebar and click 'Run Simulation' to start.")

        # Show charging/discharging explanation based on PCM type
        st.subheader("Charging/Discharging Logic")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Hot PCM (Heat Storage)**")
            st.markdown("""
            - **Charging:** Hot water heats and melts PCM (stores heat)
            - **Discharging:** Cold water cools and solidifies PCM (releases heat)
            """)

        with col2:
            st.markdown("**Cold PCM (Cold Storage)**")
            st.markdown("""
            - **Charging:** Cold water cools and solidifies PCM (stores cold)
            - **Discharging:** Warm water heats and melts PCM (releases cold)
            """)


if __name__ == "__main__":
    main()
