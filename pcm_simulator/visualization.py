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
                           T_water_hot: float,
                           T_water_cold: float,
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

    # Reference lines
    fig.add_hline(y=T_water_hot, line_dash="dot", line_color="red",
                  annotation_text="T_hot", annotation_position="right")
    fig.add_hline(y=T_water_cold, line_dash="dot", line_color="blue",
                  annotation_text="T_cold", annotation_position="right")

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
        y=data['Q_total_kW'],
        mode='lines',
        name='Q_total',
        line=dict(color='black', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=data['Q_fin_kW'],
        mode='lines',
        name='Q_fin',
        line=dict(color='green', width=1.5, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=data['time_min'],
        y=data['Q_tube_kW'],
        mode='lines',
        name='Q_tube',
        line=dict(color='purple', width=1.5, dash='dot')
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
                             T_water_hot: float,
                             T_water_cold: float,
                             delta_max_mm: float,
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

    # Power plot (1,2)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['Q_total_kW'],
        mode='lines', name='Q_total',
        line=dict(color='black', width=2)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=data['time_min'], y=data['Q_fin_kW'],
        mode='lines', name='Q_fin',
        line=dict(color='green', width=1.5, dash='dash')
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


def export_plot_to_png(fig: go.Figure, filename: str, width: int = 1200, height: int = 800) -> bytes:
    """Export plot to PNG bytes"""
    return fig.to_image(format="png", width=width, height=height)


def export_data_to_csv(data: Dict[str, np.ndarray]) -> str:
    """Export data to CSV string"""
    import pandas as pd
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
    })
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
