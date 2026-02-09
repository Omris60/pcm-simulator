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
    SimulationMode
)
from visualization import (
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

        st.divider()

        # ----- Operating Conditions -----
        st.subheader("3. Operating Conditions")

        Q_water = st.number_input("Water Flow Rate (lpm)", value=20.0, min_value=1.0, max_value=500.0)

        # Set default temperatures based on PCM category
        if pcm.category == PCMCategory.HOT:
            default_T_hot = pcm.T_liquidus + 3
            default_T_cold = pcm.T_solidus - 4
            default_T_init = pcm.T_solidus - 4  # Start cold for charging
        else:  # COLD
            default_T_hot = pcm.T_liquidus + 5
            default_T_cold = pcm.T_solidus - 5
            default_T_init = pcm.T_liquidus + 3  # Start warm for charging

        col1, col2 = st.columns(2)
        with col1:
            T_hot = st.number_input("Hot Water Temp (C)", value=float(default_T_hot), min_value=-10.0, max_value=100.0)
        with col2:
            T_cold = st.number_input("Cold Water Temp (C)", value=float(default_T_cold), min_value=-10.0, max_value=100.0)

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
            t_max = st.number_input("Max Time (min)", value=120, min_value=10, max_value=600)
        with col2:
            dt = st.number_input("Time Step (s)", value=1.0, min_value=0.1, max_value=10.0)

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
        T_water_cold=T_cold
    )

    config = SimulationConfig(
        dt=dt,
        t_max=t_max * 60,  # Convert to seconds
        T_pcm_initial=T_pcm_init
    )

    # System info columns
    col1, col2, col3, col4 = st.columns(4)

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

    with col1:
        st.metric("PCM Mass", f"{m_pcm:.1f} kg")
    with col2:
        st.metric("PCM Volume", f"{V_pcm*1000:.1f} L")
    with col3:
        st.metric("Latent Capacity", f"{E_latent:.2f} kWh")
    with col4:
        st.metric("Pressure Drop", f"{delta_P_kPa:.1f} kPa")

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
                results, summary = sim.run_full_cycle(t_max * 30, t_max * 30)  # Half time each

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

        st.divider()

        # Summary metrics
        st.subheader("Results Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Time", f"{summary.total_time_min:.1f} min")
        with col2:
            st.metric("Energy", f"{summary.total_energy_kWh:.2f} kWh")
        with col3:
            st.metric("Avg Power", f"{summary.average_power_kW:.2f} kW")
        with col4:
            st.metric("Final T_PCM", f"{summary.final_T_pcm:.1f} C")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Final Phase", summary.final_phase)
        with col2:
            st.metric("Melt Fraction", f"{summary.final_melt_fraction*100:.0f}%")

        st.divider()

        # Plots
        st.subheader("Simulation Plots")

        # Plot selection
        plot_type = st.radio(
            "View",
            ["Dashboard", "Individual Plots"],
            horizontal=True
        )

        if plot_type == "Dashboard":
            fig = create_combined_dashboard(
                data,
                T_water_hot=T_hot,
                T_water_cold=T_cold,
                delta_max_mm=hex_geom.delta_max_fin * 1000,
                title=f"PCM Simulation: {pcm.name} - {sim_mode}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Individual plots in tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Temperature", "Power", "Energy", "Melt Fraction", "Front Position", "Enthalpy"
            ])

            with tab1:
                fig = create_temperature_plot(data, T_hot, T_cold)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig = create_power_plot(data)
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                fig = create_energy_plot(data)
                st.plotly_chart(fig, use_container_width=True)

            with tab4:
                fig = create_melt_fraction_plot(data)
                st.plotly_chart(fig, use_container_width=True)

            with tab5:
                fig = create_front_position_plot(data, hex_geom.delta_max_fin * 1000)
                st.plotly_chart(fig, use_container_width=True)

            with tab6:
                fig = create_enthalpy_plot(data)
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
