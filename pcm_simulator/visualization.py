"""
Visualization Module
====================
Plotly-based interactive plotting and export functions.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional
import io


def create_temperature_plot(data: Dict[str, np.ndarray],
                           T_water_hot: Optional[float] = None,
                           T_water_cold: Optional[float] = None,
                           title: str = "Temperature vs Time") -> go.Figure:
    """Create temperature vs time plot"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=data['T_pcm_C'],
        mode='lines',
        name='T_PCM',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=data['T_water_out_C'],
        mode='lines',
        name='T_water_out',
        line=dict(color='red', width=1.5, dash='dash')
    ))

    # Reference lines (only when constant temperature mode)
    if T_water_hot is not None:
        fig.add_hline(y=T_water_hot, line_dash="dot", line_color="red",
                      annotation_text="T_hot", annotation_position="right")
    if T_water_cold is not None:
        fig.add_hline(y=T_water_cold, line_dash="dot", line_color="blue",
                      annotation_text="T_cold", annotation_position="right")

    # T_water_in trace (for closed-loop mode where it varies)
    if 'T_water_in_C' in data and np.ptp(data['T_water_in_C']) > 0.01:
        fig.add_trace(go.Scatter(
            x=data['time_min'],
            y=data['T_water_in_C'],
            mode='lines',
            name='T_water_in',
            line=dict(color='orange', width=1.5, dash='dot')
        ))

    # T_tank trace (for closed-loop mode)
    if 'T_tank_C' in data and np.any(data['T_tank_C'] != 0):
        fig.add_trace(go.Scatter(
            x=data['time_min'],
            y=data['T_tank_C'],
            mode='lines',
            name='T_tank',
            line=dict(color='green', width=1.5, dash='dashdot')
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Temperature (C)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )

    return fig


def create_power_plot(data: Dict[str, np.ndarray],
                     title: str = "Heat Transfer Rate vs Time") -> go.Figure:
    """Create power vs time plot"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=np.abs(data['Q_total_kW']),
        mode='lines',
        name='Q_total',
        line=dict(color='black', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=np.abs(data['Q_fin_kW']),
        mode='lines',
        name='Q_fin',
        line=dict(color='green', width=1.5, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=np.abs(data['Q_tube_kW']),
        mode='lines',
        name='Q_tube',
        line=dict(color='purple', width=1.5, dash='dot')
    ))

    # Wall heat loss trace (only when non-zero)
    if 'Q_loss_kW' in data and np.any(data['Q_loss_kW'] != 0):
        fig.add_trace(go.Scatter(
            x=data['time_min'],
            y=np.abs(data['Q_loss_kW']),
            mode='lines',
            name='Q_loss (wall)',
            line=dict(color='brown', width=1.5, dash='dashdot')
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Power (kW)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )

    return fig


def create_energy_plot(data: Dict[str, np.ndarray],
                      title: str = "Cumulative Energy vs Time") -> go.Figure:
    """Create energy vs time plot"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=data['E_total_kWh'],
        mode='lines',
        name='Energy',
        line=dict(color='blue', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 100, 255, 0.2)'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Energy (kWh)",
        hovermode='x unified'
    )

    return fig


def create_melt_fraction_plot(data: Dict[str, np.ndarray],
                             title: str = "Melt Fraction vs Time") -> go.Figure:
    """Create melt fraction vs time plot"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=data['f_melted'] * 100,
        mode='lines',
        name='Melt Fraction',
        line=dict(color='orange', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.2)'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Melt Fraction (%)",
        yaxis=dict(range=[0, 105]),
        hovermode='x unified'
    )

    return fig


def create_front_position_plot(data: Dict[str, np.ndarray],
                               delta_max_mm: float,
                               title: str = "Front Position vs Time") -> go.Figure:
    """Create front position vs time plot"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=data['delta_fin_mm'],
        mode='lines',
        name='Fin Front',
        line=dict(color='green', width=2)
    ))

    fig.add_hline(y=delta_max_mm, line_dash="dot", line_color="green",
                  annotation_text="Max", annotation_position="right")

    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Front Position (mm)",
        hovermode='x unified'
    )

    return fig


def create_enthalpy_plot(data: Dict[str, np.ndarray],
                        title: str = "Specific Enthalpy vs Time") -> go.Figure:
    """Create enthalpy vs time plot"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=data['H_specific_kJ_kg'],
        mode='lines',
        name='H_specific',
        line=dict(color='purple', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Specific Enthalpy (kJ/kg)",
        hovermode='x unified'
    )

    return fig


def create_combined_dashboard(data: Dict[str, np.ndarray],
                             T_water_hot: Optional[float] = None,
                             T_water_cold: Optional[float] = None,
                             delta_max_mm: float = 0.0,
                             title: str = "PCM Simulation Results") -> go.Figure:
    """Create combined dashboard with all plots"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Temperature vs Time',
            'Heat Transfer Rate vs Time',
            'Cumulative Energy vs Time',
            'Melt Fraction vs Time',
            'Front Position vs Time',
            'Specific Enthalpy vs Time'
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )

    # Temperature plot (1,1)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['T_pcm_C'],
        mode='lines', name='T_PCM',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['T_water_out_C'],
        mode='lines', name='T_water_out',
        line=dict(color='red', width=1.5, dash='dash')
    ), row=1, col=1)

    # T_water_in trace (for closed-loop mode)
    if 'T_water_in_C' in data and np.ptp(data['T_water_in_C']) > 0.01:
        fig.add_trace(go.Scatter(
            x=data['time_min'], y=data['T_water_in_C'],
            mode='lines', name='T_water_in',
            line=dict(color='orange', width=1.5, dash='dot')
        ), row=1, col=1)

    # T_tank trace (for closed-loop mode)
    if 'T_tank_C' in data and np.any(data['T_tank_C'] != 0):
        fig.add_trace(go.Scatter(
            x=data['time_min'], y=data['T_tank_C'],
            mode='lines', name='T_tank',
            line=dict(color='green', width=1.5, dash='dashdot')
        ), row=1, col=1)

    # Power plot (1,2)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=np.abs(data['Q_total_kW']),
        mode='lines', name='Q_total',
        line=dict(color='black', width=2)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=np.abs(data['Q_fin_kW']),
        mode='lines', name='Q_fin',
        line=dict(color='green', width=1.5, dash='dash')
    ), row=1, col=2)

    # Wall heat loss trace (only when non-zero)
    if 'Q_loss_kW' in data and np.any(data['Q_loss_kW'] != 0):
        fig.add_trace(go.Scatter(
            x=data['time_min'], y=np.abs(data['Q_loss_kW']),
            mode='lines', name='Q_loss (wall)',
            line=dict(color='brown', width=1.5, dash='dashdot')
        ), row=1, col=2)

    # Energy plot (2,1)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['E_total_kWh'],
        mode='lines', name='Energy',
        line=dict(color='blue', width=2),
        fill='tozeroy', fillcolor='rgba(0, 100, 255, 0.2)'
    ), row=2, col=1)

    # Melt fraction plot (2,2)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['f_melted'] * 100,
        mode='lines', name='Melt Fraction',
        line=dict(color='orange', width=2),
        fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.2)'
    ), row=2, col=2)

    # Front position plot (3,1)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['delta_fin_mm'],
        mode='lines', name='Fin Front',
        line=dict(color='green', width=2)
    ), row=3, col=1)

    # Enthalpy plot (3,2)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['H_specific_kJ_kg'],
        mode='lines', name='H_specific',
        line=dict(color='purple', width=2)
    ), row=3, col=2)

    # Update axes labels
    fig.update_xaxes(title_text="Time (min)", row=3, col=1)
    fig.update_xaxes(title_text="Time (min)", row=3, col=2)

    fig.update_yaxes(title_text="Temperature (C)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Energy (kWh)", row=2, col=1)
    fig.update_yaxes(title_text="Melt Fraction (%)", row=2, col=2)
    fig.update_yaxes(title_text="Position (mm)", row=3, col=1)
    fig.update_yaxes(title_text="Enthalpy (kJ/kg)", row=3, col=2)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=900,
        showlegend=False,
        hovermode='x unified'
    )

    return fig


def create_wall_loss_plot(data: Dict[str, np.ndarray],
                          title: str = "Wall Heat Loss vs Time") -> go.Figure:
    """Create wall heat loss vs time plot (positive values)"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=np.abs(data['Q_loss_kW']),
        mode='lines',
        name='Q_loss (wall)',
        line=dict(color='brown', width=2),
        fill='tozeroy',
        fillcolor='rgba(139, 69, 19, 0.2)'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Wall Heat Loss (kW)",
        hovermode='x unified'
    )

    return fig


def create_loop_temperature_plot(data: Dict[str, np.ndarray],
                                 title: str = "Loop Temperature Sensors vs Time") -> go.Figure:
    """Create plot showing all 6 temperature sensors around the water loop."""
    fig = go.Figure()

    # T_pcm as reference dashed line
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['T_pcm_C'],
        mode='lines', name='T_PCM (ref)',
        line=dict(color='gray', width=1.5, dash='dash')
    ))

    # 1. Tank
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['T_tank_C'],
        mode='lines', name='T_tank',
        line=dict(color='#1565C0', width=2)
    ))

    # 2. At source inlet (after Pipe 1 loss)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['T_at_source_inlet_C'],
        mode='lines', name='T_at_source_inlet',
        line=dict(color='#42A5F5', width=1.5, dash='dot')
    ))

    # 3. After source (before pump)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['T_after_source_C'],
        mode='lines', name='T_after_source',
        line=dict(color='#E65100', width=2)
    ))

    # 3b. After pump (between source and Pipe 2)
    if 'T_after_pump_C' in data and np.any(data['T_after_pump_C'] != 0):
        fig.add_trace(go.Scatter(
            x=data['time_min'], y=data['T_after_pump_C'],
            mode='lines', name='T_after_pump',
            line=dict(color='#D81B60', width=2, dash='dashdot')
        ))

    # 4. Water in at HEX (after Pipe 2 loss)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['T_water_in_C'],
        mode='lines', name='T_water_in (HEX)',
        line=dict(color='#FB8C00', width=1.5, dash='dot')
    ))

    # 5. Water out of HEX
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['T_water_out_C'],
        mode='lines', name='T_water_out (HEX)',
        line=dict(color='#2E7D32', width=2)
    ))

    # 6. Return to tank (after Pipe 3 loss)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['T_return_to_tank_C'],
        mode='lines', name='T_return_to_tank',
        line=dict(color='#66BB6A', width=1.5, dash='dot')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Temperature (°C)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )

    return fig


def create_pipe_loss_plot(data: Dict[str, np.ndarray],
                          title: str = "Pipe Heat Loss vs Time") -> go.Figure:
    """Create plot showing per-segment and total pipe heat losses."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=np.abs(data['Q_pipe_loss_1_kW']),
        mode='lines',
        name='Pipe 1: Tank → Source',
        line=dict(color='#1565C0', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=np.abs(data['Q_pipe_loss_2_kW']),
        mode='lines',
        name='Pipe 2: Source → HEX',
        line=dict(color='#E65100', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=np.abs(data['Q_pipe_loss_3_kW']),
        mode='lines',
        name='Pipe 3: HEX → Tank',
        line=dict(color='#2E7D32', width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=np.abs(data['Q_pipe_loss_kW']),
        mode='lines',
        name='Total Pipe Loss',
        line=dict(color='black', width=2, dash='dash')
    ))

    # Pump heat trace (when present and non-zero)
    if 'Q_pump_kW' in data and np.any(data['Q_pump_kW'] != 0):
        fig.add_trace(go.Scatter(
            x=data['time_min'],
            y=data['Q_pump_kW'],
            mode='lines',
            name='Pump Heat',
            line=dict(color='#D81B60', width=2, dash='dashdot')
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Pipe Heat Loss (kW)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )

    return fig


def create_source_power_plot(data: Dict[str, np.ndarray],
                             title: str = "Source/Sink Power vs Time") -> go.Figure:
    """Create source/sink power vs time plot (for closed-loop mode)"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=data['Q_source_kW'],
        mode='lines',
        name='Q_source',
        line=dict(color='darkorange', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 140, 0, 0.2)'
    ))

    fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Source Power (kW)",
        hovermode='x unified'
    )

    return fig


def export_plot_to_png(fig: go.Figure, filename: str, width: int = 1200, height: int = 800) -> bytes:
    """Export plot to PNG bytes"""
    return fig.to_image(format="png", width=width, height=height)


def export_data_to_csv(data: Dict[str, np.ndarray]) -> str:
    """Export data to CSV string"""
    import pandas as pd
    columns = {
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
    }
    # Add closed-loop columns when present and non-trivial
    if 'T_water_in_C' in data and np.ptp(data['T_water_in_C']) > 0.01:
        columns['T_water_in (C)'] = data['T_water_in_C']
    if 'T_tank_C' in data and np.any(data['T_tank_C'] != 0):
        columns['T_tank (C)'] = data['T_tank_C']
    if 'Q_source_kW' in data and np.any(data['Q_source_kW'] != 0):
        columns['Q_source (kW)'] = data['Q_source_kW']
    if 'Q_loss_kW' in data and np.any(data['Q_loss_kW'] != 0):
        columns['Q_loss (kW)'] = data['Q_loss_kW']
    if 'E_loss_kWh' in data and np.any(data['E_loss_kWh'] != 0):
        columns['E_loss (kWh)'] = data['E_loss_kWh']
    if 'Q_pipe_loss_kW' in data and np.any(data['Q_pipe_loss_kW'] != 0):
        columns['Q_pipe_loss (kW)'] = data['Q_pipe_loss_kW']
    if 'E_pipe_loss_kWh' in data and np.any(data['E_pipe_loss_kWh'] != 0):
        columns['E_pipe_loss (kWh)'] = data['E_pipe_loss_kWh']
    # Loop temperature sensors (closed-loop mode)
    if 'T_at_source_inlet_C' in data and np.any(data['T_at_source_inlet_C'] != 0):
        columns['T_at_source_inlet (C)'] = data['T_at_source_inlet_C']
    if 'T_after_source_C' in data and np.any(data['T_after_source_C'] != 0):
        columns['T_after_source (C)'] = data['T_after_source_C']
    if 'T_return_to_tank_C' in data and np.any(data['T_return_to_tank_C'] != 0):
        columns['T_return_to_tank (C)'] = data['T_return_to_tank_C']
    # Per-segment pipe losses
    if 'Q_pipe_loss_1_kW' in data and np.any(data['Q_pipe_loss_1_kW'] != 0):
        columns['Q_pipe_loss_1 (kW)'] = data['Q_pipe_loss_1_kW']
    if 'Q_pipe_loss_2_kW' in data and np.any(data['Q_pipe_loss_2_kW'] != 0):
        columns['Q_pipe_loss_2 (kW)'] = data['Q_pipe_loss_2_kW']
    if 'Q_pipe_loss_3_kW' in data and np.any(data['Q_pipe_loss_3_kW'] != 0):
        columns['Q_pipe_loss_3 (kW)'] = data['Q_pipe_loss_3_kW']
    # Pump heat
    if 'T_after_pump_C' in data and np.any(data['T_after_pump_C'] != 0):
        columns['T_after_pump (C)'] = data['T_after_pump_C']
    if 'Q_pump_kW' in data and np.any(data['Q_pump_kW'] != 0):
        columns['Q_pump (kW)'] = data['Q_pump_kW']
    df = pd.DataFrame(columns)
    return df.to_csv(index=False)


def create_enthalpy_curve_plot(pcm_material) -> go.Figure:
    """Create plot showing PCM enthalpy curves"""
    fig = go.Figure()

    temps = pcm_material.enthalpy_data.temperatures
    dH_melt = pcm_material.enthalpy_data.dH_melting
    dH_solid = pcm_material.enthalpy_data.dH_solidifying

    fig.add_trace(go.Bar(
        x=temps[:-1] + 0.5,
        y=dH_melt[:-1],
        name='Melting',
        marker_color='red',
        opacity=0.7,
        width=0.4,
        offset=-0.2
    ))

    fig.add_trace(go.Bar(
        x=temps[:-1] + 0.5,
        y=dH_solid[:-1],
        name='Solidifying',
        marker_color='blue',
        opacity=0.7,
        width=0.4,
        offset=0.2
    ))

    fig.update_layout(
        title=f"Enthalpy Distribution: {pcm_material.name}",
        xaxis_title="Temperature (C)",
        yaxis_title="dH/dT (kJ/kg/C)",
        barmode='group',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    return fig


def create_cumulative_enthalpy_plot(pcm_material) -> go.Figure:
    """Create plot showing cumulative enthalpy curves"""
    fig = go.Figure()

    temps = pcm_material.enthalpy_data.temperatures
    H_melt = pcm_material.enthalpy_data.H_melting
    H_solid = pcm_material.enthalpy_data.H_solidifying

    fig.add_trace(go.Scatter(
        x=temps,
        y=H_melt,
        mode='lines+markers',
        name='Melting',
        line=dict(color='red', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=temps,
        y=H_solid,
        mode='lines+markers',
        name='Solidifying',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title=f"Cumulative Enthalpy: {pcm_material.name}",
        xaxis_title="Temperature (C)",
        yaxis_title="H (kJ/kg)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


# =============================================================================
# SYSTEM SCHEMATIC (PFD)
# =============================================================================

def _schematic_box(fig, x0, y0, x1, y1, fill, border):
    """Add a rectangle shape to the schematic."""
    fig.add_shape(
        type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(color=border, width=2),
        fillcolor=fill, layer="below",
    )


def _schematic_label(fig, x, y, text, size=13, color="#333"):
    """Add a text label (no arrow) to the schematic."""
    fig.add_annotation(
        x=x, y=y, text=text, showarrow=False,
        font=dict(size=size, color=color, family="Arial"),
        align="center",
    )


def _schematic_arrow(fig, x0, y0, x1, y1, color="#1565C0", width=2.5, label=""):
    """Add an arrow with optional midpoint label."""
    fig.add_annotation(
        x=x1, y=y1, ax=x0, ay=y0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1.3,
        arrowwidth=width, arrowcolor=color,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        fig.add_annotation(
            x=mx, y=my + 0.04, text=label, showarrow=False,
            font=dict(size=11, color=color, family="Arial"),
        )


def _pcm_cell_labels(fig, cx, pcm, box, hex_geom, m_pcm, E_latent, y_top=0.72):
    """Draw the PCM Cell box labels at horizontal centre cx."""
    _schematic_label(fig, cx, y_top, "<b>PCM Cell</b>", size=16, color="#2E7D32")
    _schematic_label(fig, cx, y_top - 0.10, pcm.name, size=12, color="#555")
    _schematic_label(fig, cx, y_top - 0.18,
                     f"{pcm.T_solidus:.0f}–{pcm.T_liquidus:.0f} °C", size=11, color="#777")
    _schematic_label(fig, cx, y_top - 0.25,
                     f"{box.length:.0f} × {box.width:.0f} × {box.height:.0f} mm",
                     size=10, color="#888")
    _schematic_label(fig, cx, y_top - 0.31,
                     f"A_hex = {hex_geom.A_total:.1f} m²  |  m = {m_pcm:.0f} kg",
                     size=10, color="#888")
    _schematic_label(fig, cx, y_top - 0.38,
                     f"E_latent = {E_latent:.2f} kWh", size=11, color="#555")


def _schematic_layout(fig, title):
    """Apply common layout settings for schematics."""
    fig.update_layout(
        xaxis=dict(range=[0, 1], visible=False, fixedrange=True),
        yaxis=dict(range=[0, 1], visible=False, fixedrange=True, scaleanchor="x"),
        width=800, height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(text=title, font=dict(size=15)),
    )


def create_system_schematic(pcm, box, hex_geom, operating, m_pcm, E_latent,
                            pipe_loss_config=None, pump_config=None) -> go.Figure:
    """Create a PFD-style system schematic.

    Draws either a constant-temperature or closed-loop (heat source/sink)
    diagram depending on operating.water_supply_mode.
    """
    from simulation_core import WaterSupplyMode

    is_source_sink = (operating.water_supply_mode == WaterSupplyMode.HEAT_SOURCE_SINK)
    flow = f"{operating.Q_water_lpm:.0f} lpm"

    if is_source_sink:
        return _schematic_source_sink(pcm, box, hex_geom, operating, m_pcm, E_latent, flow,
                                       pipe_loss_config=pipe_loss_config,
                                       pump_config=pump_config)
    else:
        return _schematic_constant_temp(pcm, box, hex_geom, operating, m_pcm, E_latent, flow)


def _schematic_constant_temp(pcm, box, hex_geom, operating, m_pcm, E_latent, flow):
    fig = go.Figure()

    # Hot Water Supply
    _schematic_box(fig, 0.02, 0.60, 0.30, 0.90, "rgba(239,83,80,0.12)", "#E53935")
    _schematic_label(fig, 0.16, 0.80, "<b>Hot Water</b><br><b>Supply</b>", size=14, color="#C62828")
    _schematic_label(fig, 0.16, 0.65, f"T_hot = {operating.T_water_hot:.0f} °C", size=11, color="#C62828")

    # Cold Water Supply
    _schematic_box(fig, 0.02, 0.10, 0.30, 0.40, "rgba(66,165,245,0.12)", "#1E88E5")
    _schematic_label(fig, 0.16, 0.30, "<b>Cold Water</b><br><b>Supply</b>", size=14, color="#1565C0")
    _schematic_label(fig, 0.16, 0.15, f"T_cold = {operating.T_water_cold:.0f} °C", size=11, color="#1565C0")

    # PCM Cell
    _schematic_box(fig, 0.55, 0.22, 0.98, 0.78, "rgba(102,187,106,0.12)", "#43A047")
    _pcm_cell_labels(fig, 0.765, pcm, box, hex_geom, m_pcm, E_latent, y_top=0.68)

    # Arrows with temperature labels
    _schematic_arrow(fig, 0.30, 0.75, 0.55, 0.60, color="#E53935", label=flow)
    _schematic_label(fig, 0.32, 0.78, f"{operating.T_water_hot:.0f} °C", size=9, color="#C62828")

    _schematic_arrow(fig, 0.30, 0.25, 0.55, 0.40, color="#1E88E5", label=flow)
    _schematic_label(fig, 0.32, 0.22, f"{operating.T_water_cold:.0f} °C", size=9, color="#1565C0")

    # Mode hint
    _schematic_label(fig, 0.42, 0.50, "Charge ↑<br>Discharge ↓", size=10, color="#999")

    _schematic_layout(fig, "System Schematic — Constant Temperature Mode")
    return fig


def _schematic_source_sink(pcm, box, hex_geom, operating, m_pcm, E_latent, flow,
                           pipe_loss_config=None, pump_config=None):
    import numpy as _np
    fig = go.Figure()
    cfg = operating.heat_source_config

    pump_enabled = pump_config is not None and pump_config.enabled

    # Wider layout with more spacing between components
    # Water Tank
    _schematic_box(fig, 0.01, 0.32, 0.16, 0.78, "rgba(66,165,245,0.12)", "#1E88E5")
    _schematic_label(fig, 0.085, 0.66, "<b>Water</b><br><b>Tank</b>", size=14, color="#1565C0")
    _schematic_label(fig, 0.085, 0.48, f"{cfg.tank_volume_L:.0f} L", size=12, color="#555")
    _schematic_label(fig, 0.085, 0.38, f"T₀ = {cfg.T_tank_initial:.0f} °C", size=11, color="#777")

    # Heat Source/Sink
    _schematic_box(fig, 0.26, 0.32, 0.44, 0.78, "rgba(255,167,38,0.15)", "#FB8C00")
    _schematic_label(fig, 0.35, 0.66, "<b>Heat</b><br><b>Source/Sink</b>", size=13, color="#E65100")
    _schematic_label(fig, 0.35, 0.48, f"{cfg.power_kW:.1f} kW", size=12, color="#555")
    _schematic_label(fig, 0.35, 0.38, cfg.control_mode.value, size=11, color="#777")

    # Pump (between Source and Cell)
    if pump_enabled:
        _schematic_box(fig, 0.50, 0.40, 0.62, 0.70, "rgba(216,27,96,0.12)", "#D81B60")
        _schematic_label(fig, 0.56, 0.60, "<b>Pump</b>", size=13, color="#D81B60")
        _schematic_label(fig, 0.56, 0.48, f"{pump_config.power_W:.0f} W", size=11, color="#555")

    # PCM Cell
    _schematic_box(fig, 0.70, 0.24, 0.99, 0.86, "rgba(102,187,106,0.12)", "#43A047")
    _pcm_cell_labels(fig, 0.845, pcm, box, hex_geom, m_pcm, E_latent, y_top=0.76)

    # === Pre-compute design temperatures for labels ===
    T_tank = cfg.T_tank_initial

    # Estimate Pipe 1 loss (T₁ → T₂)
    pipe_enabled = pipe_loss_config is not None and pipe_loss_config.enabled
    dT_pipe1, dT_pipe2, dT_pipe3 = 0.0, 0.0, 0.0
    Q_pipe1, Q_pipe2, Q_pipe3 = 0.0, 0.0, 0.0
    water_cp = 4180
    m_dot = operating.Q_water_lpm / 60 * 1000 / 1000  # kg/s
    C_w = m_dot * water_cp

    # Estimate pump dT
    dT_pump = 0.0
    if pump_enabled and C_w > 0:
        dT_pump = pump_config.power_W / C_w

    if pipe_enabled and C_w > 0:
        pl = pipe_loss_config
        r_pipe = (pl.pipe_OD_mm / 2) / 1000
        r_outer = r_pipe + pl.insulation_thickness_mm / 1000

        def _calc_UA(L):
            if L <= 0:
                return 0.0
            R_ins = 0.0
            if pl.insulation_thickness_mm > 0 and pl.k_insulation > 0:
                R_ins = _np.log(r_outer / r_pipe) / (2 * _np.pi * pl.k_insulation * L)
            R_ext = 1.0 / (2 * _np.pi * r_outer * L * pl.h_ext)
            return 1.0 / (R_ins + R_ext)

        UA1 = _calc_UA(pl.L_tank_to_source_m)
        UA2 = _calc_UA(pl.L_source_to_hex_m)
        UA3 = _calc_UA(pl.L_hex_to_tank_m)

        Q_pipe1 = UA1 * (T_tank - pl.T_ambient)
        dT_pipe1 = Q_pipe1 / C_w if C_w > 0 else 0
        T_at_source = T_tank - dT_pipe1
        T_after_source_est = T_at_source + cfg.power_kW * 1000 / C_w if C_w > 0 else T_at_source
        T_after_pump_est = T_after_source_est + dT_pump
        Q_pipe2 = UA2 * (T_after_pump_est - pl.T_ambient)
        dT_pipe2 = Q_pipe2 / C_w if C_w > 0 else 0
        Q_pipe3 = UA3 * (T_tank - pl.T_ambient)
        dT_pipe3 = Q_pipe3 / C_w if C_w > 0 else 0

    # === Forward arrows: Tank → Source → [Pump] → Cell (top path) ===
    arrow_y = 0.70

    # Pipe 1: Tank → Source
    _schematic_arrow(fig, 0.16, arrow_y, 0.26, arrow_y, color="#1565C0", label=flow)
    _schematic_label(fig, 0.17, arrow_y + 0.06, f"T₁={T_tank:.0f}°C", size=9, color="#1565C0")
    if pipe_enabled and abs(dT_pipe1) > 0.05:
        _schematic_label(fig, 0.21, arrow_y - 0.06,
                         f"<i>ΔT≈{dT_pipe1:.1f}°C</i>", size=8, color="#9E9E9E")

    if pump_enabled:
        # Source → Pump
        _schematic_arrow(fig, 0.44, arrow_y, 0.50, arrow_y, color="#E65100")
        # Pump → Cell (Pipe 2)
        _schematic_arrow(fig, 0.62, arrow_y, 0.70, arrow_y, color="#D81B60")
        if pipe_enabled and abs(dT_pipe2) > 0.05:
            _schematic_label(fig, 0.66, arrow_y - 0.06,
                             f"<i>ΔT≈{dT_pipe2:.1f}°C</i>", size=8, color="#9E9E9E")
    else:
        # Source → Cell (Pipe 2) — no pump
        _schematic_arrow(fig, 0.44, arrow_y, 0.70, arrow_y, color="#E65100")
        if pipe_enabled and abs(dT_pipe2) > 0.05:
            _schematic_label(fig, 0.57, arrow_y - 0.06,
                             f"<i>ΔT≈{dT_pipe2:.1f}°C</i>", size=8, color="#9E9E9E")

    # === Return path: Cell → down → horizontal → up into Tank ===
    ret_y_top = 0.24
    ret_y_bot = 0.14

    # Cell exit → down
    _schematic_arrow(fig, 0.845, ret_y_top, 0.845, ret_y_bot, color="#78909C", width=2)
    # Horizontal return line
    fig.add_shape(
        type="line", x0=0.085, y0=ret_y_bot, x1=0.845, y1=ret_y_bot,
        line=dict(color="#78909C", width=2),
    )
    # Up into tank
    _schematic_arrow(fig, 0.085, ret_y_bot, 0.085, 0.32, color="#78909C", width=2)

    # Pipe 3 label on return path
    _schematic_label(fig, 0.46, ret_y_bot + 0.05, "Return", size=10, color="#78909C")
    if pipe_enabled and abs(dT_pipe3) > 0.05:
        _schematic_label(fig, 0.46, ret_y_bot - 0.05,
                         f"<i>Pipe 3: ΔT≈{dT_pipe3:.1f}°C</i>", size=8, color="#9E9E9E")

    # Annotations (bottom-right)
    annotations = []
    if pipe_enabled:
        total_Q = Q_pipe1 + Q_pipe2 + Q_pipe3
        annotations.append(f"Pipe loss ≈ {total_Q:.0f} W")
    if pump_enabled:
        annotations.append(f"Pump ΔT ≈ {dT_pump:.1f} °C")
    if annotations:
        _schematic_label(fig, 0.85, 0.05,
                         f"<i>{' | '.join(annotations)}</i>", size=9, color="#9E9E9E")

    _schematic_layout(fig, "System Schematic — Heat Source/Sink Mode")
    return fig


# =============================================================================
# PARAMETRIC SWEEP PLOTS
# =============================================================================

def create_sweep_heatmap(flow_rates, source_powers, metric_matrix,
                         metric_name, metric_unit, title) -> go.Figure:
    """Create a 2D heatmap for parametric sweep results.

    Args:
        flow_rates: 1D array of flow rates (Y axis)
        source_powers: 1D array of source powers (X axis)
        metric_matrix: 2D array [len(flow_rates) x len(source_powers)]
        metric_name: Name of the metric for hover
        metric_unit: Unit string
        title: Plot title
    """
    fig = go.Figure(data=go.Heatmap(
        z=metric_matrix,
        x=source_powers,
        y=flow_rates,
        colorscale='Viridis',
        colorbar=dict(title=f"{metric_name} ({metric_unit})"),
        hovertemplate=(
            f"Source Power: %{{x:.1f}} kW<br>"
            f"Flow Rate: %{{y:.1f}} lpm<br>"
            f"{metric_name}: %{{z:.3f}} {metric_unit}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Source Power (kW)",
        yaxis_title="Flow Rate (lpm)",
        height=500,
    )

    return fig


def create_sweep_line_plot(sweep_values, metrics_dict, x_label, title) -> go.Figure:
    """Create a 1D line plot for parametric sweep results.

    Args:
        sweep_values: 1D array of sweep parameter values (X axis)
        metrics_dict: dict of {metric_name: (values_array, unit_str)}
        x_label: Label for X axis
        title: Plot title
    """
    from plotly.subplots import make_subplots

    metric_names = list(metrics_dict.keys())
    n = len(metric_names)

    fig = make_subplots(rows=n, cols=1,
                        subplot_titles=metric_names,
                        vertical_spacing=0.08,
                        shared_xaxes=True)

    colors = ['#1565C0', '#E65100', '#2E7D32', '#7B1FA2']

    for i, name in enumerate(metric_names):
        values, unit = metrics_dict[name]
        fig.add_trace(go.Scatter(
            x=sweep_values,
            y=values,
            mode='lines+markers',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6),
        ), row=i+1, col=1)
        fig.update_yaxes(title_text=f"{name} ({unit})", row=i+1, col=1)

    fig.update_xaxes(title_text=x_label, row=n, col=1)
    fig.update_layout(
        title=title,
        height=250 * n,
        showlegend=False,
    )

    return fig
