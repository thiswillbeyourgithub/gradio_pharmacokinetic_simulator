"""
Pharmacokinetic Simulation Web UI

A Gradio-based web interface for simulating plasma drug
concentrations over time
with multiple dosing regimens. This tool demonstrates drug
accumulation patterns
based on pharmacokinetic parameters.

Created with assistance from aider.chat
"""

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for web deployment


def calculate_concentration_profile(
    dose: float,
    clearance: float,
    volume_distribution: float,
    half_life: float,
    dosing_interval: float,
    num_doses: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
        Calculate plasma concentration over time for multiple dose
    regimen.

        Uses first-order elimination kinetics and superposition
    principle to model
        drug accumulation. Each dose is treated as an independent IV
    bolus with
        exponential decay.

        Parameters
        ----------
        dose : float
            Single dose amount (mg)
        clearance : float
            Drug clearance (L/h)
        volume_distribution : float
            Volume of distribution (L)
        half_life : float
            Elimination half-life (hours)
        dosing_interval : float
            Time between doses (hours)
        num_doses : int, optional
            Number of doses to simulate. If None, calculated based on
    simulation time.

        Returns
        -------
        time_points : np.ndarray
            Time points for simulation (hours)
        concentrations : np.ndarray
            Plasma concentrations at each time point (mg/L)
    """
    # Calculate elimination rate constant from half-life
    k_elimination = np.log(2) / half_life

    # Simulation duration: 6 times the half-life to show complete elimination
    sim_duration = 6 * half_life

    # Create time points with high resolution for smooth curves
    time_points = np.linspace(0, sim_duration, 1000)

    # Calculate number of doses if not provided
    if num_doses is None:
        num_doses = int(np.ceil(sim_duration / dosing_interval)) + 1

    # Initialize concentration array
    concentrations = np.zeros_like(time_points)

    # Initial plasma concentration after IV bolus (C0 = Dose/Vd)
    c0 = dose / volume_distribution

    # Apply superposition principle for multiple doses
    for dose_number in range(num_doses):
        dose_time = dose_number * dosing_interval

        # Only consider time points after this dose is given
        mask = time_points >= dose_time
        time_since_dose = time_points[mask] - dose_time

        # Add exponential decay from this dose to total concentration
        concentrations[mask] += c0 * np.exp(-k_elimination * time_since_dose)

    return time_points, concentrations


def create_pk_plot(
    dose: float,
    clearance: float,
    volume_distribution: float,
    half_life: float,
    dosing_interval: float,
) -> plt.Figure:
    """
        Create pharmacokinetic concentration-time plot.

        Generates a matplotlib figure showing plasma concentration
    over time with
        multiple dosing. Includes styling and annotations for better
    interpretation.

        Parameters
        ----------
        dose : float
            Single dose amount (mg)
        clearance : float
            Drug clearance (L/h)
        volume_distribution : float
            Volume of distribution (L)
        half_life : float
            Elimination half-life (hours)
        dosing_interval : float
            Time between doses (hours)

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated plot figure
    """
    # Calculate concentration profile
    time_points, concentrations = calculate_concentration_profile(
        dose=dose,
        clearance=clearance,
        volume_distribution=volume_distribution,
        half_life=half_life,
        dosing_interval=dosing_interval,
    )

    # Create the plot with professional styling
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot concentration-time curve
    ax.plot(
        time_points, concentrations, "b-", linewidth=2, label="Plasma Concentration"
    )

    # Add vertical lines for dose administration times
    sim_duration = 6 * half_life
    num_doses = int(np.ceil(sim_duration / dosing_interval)) + 1

    for dose_num in range(num_doses):
        dose_time = dose_num * dosing_interval
        if dose_time <= sim_duration:
            ax.axvline(x=dose_time, color="red", linestyle="--", alpha=0.6, linewidth=1)
            if dose_num == 0:
                ax.axvline(
                    x=dose_time,
                    color="red",
                    linestyle="--",
                    alpha=0.6,
                    linewidth=1,
                    label="Dose Administration",
                )

    # Add vertical lines at 24-hour intervals to mark days
    day_interval = 24  # hours
    num_days = int(np.ceil(sim_duration / day_interval)) + 1

    for day_num in range(1, num_days):  # Start from day 1, skip day 0
        day_time = day_num * day_interval
        if day_time <= sim_duration:
            ax.axvline(x=day_time, color="gray", linestyle=":", alpha=0.4, linewidth=1)
            if day_num == 1:
                ax.axvline(
                    x=day_time,
                    color="gray",
                    linestyle=":",
                    alpha=0.4,
                    linewidth=1,
                    label="24h Intervals",
                )

    # Formatting and labels
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Plasma Concentration (mg/L)", fontsize=12)
    ax.set_title(
        f"Pharmacokinetic Profile\n"
        f"Dose: {dose} mg, t½: {half_life} h, τ: {dosing_interval} h",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set reasonable y-axis limits
    ax.set_ylim(0, np.max(concentrations) * 1.1)
    ax.set_xlim(0, sim_duration)

    plt.tight_layout()
    return fig


def update_plot(
    dose: float,
    clearance: float,
    volume_distribution: float,
    half_life: float,
    dosing_interval: float,
) -> plt.Figure:
    """
        Wrapper function for Gradio interface to update the plot.

        Validates input parameters and creates updated
    pharmacokinetic plot.
        Ensures all parameters are positive to prevent mathematical
    errors.

        Parameters
        ----------
        dose : float
            Single dose amount (mg)
        clearance : float
            Drug clearance (L/h)
        volume_distribution : float
            Volume of distribution (L)
        half_life : float
            Elimination half-life (hours)
        dosing_interval : float
            Time between doses (hours)

        Returns
        -------
        fig : matplotlib.figure.Figure
            Updated plot figure
    """
    # Input validation to prevent errors
    if any(
        param <= 0
        for param in [dose, clearance, volume_distribution, half_life, dosing_interval]
    ):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(
            0.5,
            0.5,
            "All parameters must be positive values",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=16,
        )
        ax.set_title("Invalid Parameters", fontsize=14)
        return fig

    return create_pk_plot(
        dose, clearance, volume_distribution, half_life, dosing_interval
    )


def create_gradio_interface() -> gr.Interface:
    """
        Create and configure the Gradio web interface.

        Sets up input components for pharmacokinetic parameters and
    output plot.
        Uses reasonable default values for a typical oral medication
    scenario.

        Returns
        -------
        iface : gradio.Interface
            Configured Gradio interface ready for launch
    """
    # Define input components with reasonable defaults and ranges
    inputs = [
        gr.Slider(
            minimum=1,
            maximum=1000,
            value=100,
            step=1,
            label="Dose (mg)",
            info="Amount of drug administered per dose",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=50,
            value=5,
            step=0.1,
            label="Clearance (L/h)",
            info="Rate of drug elimination from the body",
        ),
        gr.Slider(
            minimum=1,
            maximum=500,
            value=70,
            step=1,
            label="Volume of Distribution (L)",
            info="Apparent volume in which drug distributes",
        ),
        gr.Slider(
            minimum=0.5,
            maximum=48,
            value=8,
            step=0.5,
            label="Half-life (hours)",
            info="Time for drug concentration to decrease by 50%",
        ),
        gr.Number(
            value=12,
            minimum=0.1,
            maximum=168,
            label="Dosing Interval (hours)",
            info="Time between consecutive doses",
        ),
    ]

    # Define output component
    outputs = gr.Plot(label="Pharmacokinetic Profile")

    # Create interface with professional styling
    iface = gr.Interface(
        fn=update_plot,
        inputs=inputs,
        outputs=outputs,
        title="Pharmacokinetic Simulation Tool",
        description="""
        Simulate plasma drug concentrations over time with
multiple dosing regimens.
        Adjust the pharmacokinetic parameters and dosing interval
to see how drug
        accumulation patterns change. The simulation runs for 6
half-lives to show
        complete elimination behavior.
        """,
        theme=gr.themes.Soft(),
        allow_flagging="never",
    )

    return iface


def main() -> None:
    """
        Main function to launch the Gradio web application.

        Creates the interface and starts the web server with public
    sharing disabled
        by default for security. Enable share=True for temporary
    public links.
    """
    iface = create_gradio_interface()

    # Launch with local access (set share=True for public access)
    iface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True for public sharing
        show_error=True,
    )


if __name__ == "__main__":
    main()
