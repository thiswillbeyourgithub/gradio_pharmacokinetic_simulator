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
    absorption_rate_constant: float,
    half_life: float,
    dose_times: List[float],
    plot_duration: float = 24.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
        Calculate plasma concentration over time for multiple dose
    regimen with oral absorption.

        Uses first-order absorption and elimination kinetics with
    superposition principle
        to model drug accumulation. Each dose is treated as an
    independent oral dose with
        first-order absorption followed by first-order elimination.

        Parameters
        ----------
        dose : float
            Single dose amount (mg)
        absorption_rate_constant : float
            First-order absorption rate constant (1/h)
        half_life : float
            Elimination half-life (hours)
        dose_times : List[float]
            List of dosing times within a 24-hour period (hours, 0-24)
        plot_duration : float, optional
            Duration of simulation to plot (hours), by default 24.0
        averaging_interval : float, optional
            Time interval for computing average concentrations (hours), by default 4.0

        Returns
        -------
        time_points : np.ndarray
            Time points for simulation (hours)
        concentrations : np.ndarray
            Plasma concentrations at each time point (normalized units)
    """
    # Calculate elimination rate constant from half-life
    k_elimination = np.log(2) / half_life

    # Absorption rate constant
    ka = absorption_rate_constant

    # Use specified plot duration
    sim_duration = plot_duration

    # Create time points with high resolution for smooth curves
    time_points = np.linspace(0, sim_duration, 1000)

    # Initialize concentration array
    concentrations = np.zeros_like(time_points)

    # Generate all dose administration times across simulation duration
    all_dose_times = []
    num_days = int(np.ceil(sim_duration / 24)) + 1

    for day in range(num_days):
        for dose_time in dose_times:
            absolute_dose_time = day * 24 + dose_time
            if absolute_dose_time <= plot_duration:
                all_dose_times.append(absolute_dose_time)

    # Apply superposition principle for multiple doses
    for dose_time in all_dose_times:
        # Only consider time points after this dose is given
        mask = time_points >= dose_time
        time_since_dose = time_points[mask] - dose_time

        # First-order absorption and elimination model
        # C(t) = (Dose * ka) / (ka - ke) * (exp(-ke*t) - exp(-ka*t))
        # Assumes volume of distribution = 1L for normalized concentrations
        if ka != k_elimination:  # Avoid division by zero
            absorption_term = np.exp(-k_elimination * time_since_dose)
            elimination_term = np.exp(-ka * time_since_dose)
            concentrations[mask] += (
                (dose * ka)
                / (ka - k_elimination)
                * (absorption_term - elimination_term)
            )
        else:
            # Special case when ka = ke (flip-flop kinetics)
            concentrations[mask] += (
                dose * ka * time_since_dose * np.exp(-ka * time_since_dose)
            )

    return time_points, concentrations


def create_pk_plot(
    dose: float,
    absorption_rate_constant: float,
    half_life: float,
    dose_times: List[float],
    plot_duration: float = 24.0,
    averaging_interval: float = 4.0,
) -> plt.Figure:
    """
        Create pharmacokinetic concentration-time plot for oral absorption.

        Generates a matplotlib figure showing plasma concentration
    over time with
        multiple oral dosing. Includes styling and annotations for better
    interpretation.

        Parameters
        ----------
        dose : float
            Single dose amount (mg)
        absorption_rate_constant : float
            First-order absorption rate constant (1/h)
        half_life : float
            Elimination half-life (hours)
        dose_times : List[float]
            List of dosing times within a 24-hour period (hours, 0-24)
        plot_duration : float, optional
            Duration of simulation to plot (hours), by default 24.0

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated plot figure
    """
    # Use specified plot duration
    sim_duration = plot_duration

    # Calculate concentration profile
    time_points, concentrations = calculate_concentration_profile(
        dose=dose,
        absorption_rate_constant=absorption_rate_constant,
        half_life=half_life,
        dose_times=dose_times,
        plot_duration=plot_duration,
    )

    # Create the plot with professional styling
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot concentration-time curve
    ax.plot(
        time_points, concentrations, "b-", linewidth=2, label="Plasma Concentration"
    )

    # Calculate and plot average concentrations over specified intervals
    # Create time bins for averaging based on the configurable interval
    max_hours = int(np.ceil(plot_duration))
    interval_times = np.arange(0, max_hours + averaging_interval, averaging_interval)
    interval_averages = []

    for i in range(len(interval_times) - 1):
        start_time = interval_times[i]
        end_time = interval_times[i + 1]
        # Find all time points within this interval
        interval_mask = (time_points >= start_time) & (time_points < end_time)
        if np.any(interval_mask):
            # Calculate average concentration for this interval
            avg_conc = np.mean(concentrations[interval_mask])
            interval_averages.append(avg_conc)
        else:
            interval_averages.append(0)

    # Plot averages as a step function
    interval_plot_times = np.repeat(interval_times[:-1], 2)  # Duplicate for step effect
    interval_plot_concentrations = np.repeat(
        interval_averages, 2
    )  # Duplicate for step effect

    ax.plot(
        interval_plot_times,
        interval_plot_concentrations,
        "g-",
        linewidth=2,
        alpha=0.7,
        label=f"{averaging_interval:.1f}h Average Concentration",
    )

    # Add vertical lines for dose administration times
    num_days = int(np.ceil(plot_duration / 24)) + 1

    dose_line_added = False
    for day in range(num_days):
        for dose_time in dose_times:
            absolute_dose_time = day * 24 + dose_time
            if absolute_dose_time <= sim_duration:
                ax.axvline(
                    x=absolute_dose_time,
                    color="red",
                    linestyle="--",
                    alpha=0.6,
                    linewidth=1,
                )
                if not dose_line_added:
                    ax.axvline(
                        x=absolute_dose_time,
                        color="red",
                        linestyle="--",
                        alpha=0.6,
                        linewidth=1,
                        label="Dose Administration",
                    )
                    dose_line_added = True

    # Add vertical lines at 24-hour intervals to mark days
    day_interval = 24  # hours
    num_days = int(np.ceil(plot_duration / day_interval)) + 1

    for day_num in range(1, num_days):  # Start from day 1, skip day 0
        day_time = day_num * day_interval
        if day_time <= plot_duration:
            ax.axvline(x=day_time, color="gray", linestyle=":", alpha=0.7, linewidth=2)
            if day_num == 1:
                ax.axvline(
                    x=day_time,
                    color="gray",
                    linestyle=":",
                    alpha=0.7,
                    linewidth=2,
                    label="24h Intervals",
                )

    # Formatting and labels
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Plasma Concentration (normalized units)", fontsize=12)
    dose_times_str = ", ".join([f"{t:.1f}h" for t in dose_times])
    ax.set_title(
        f"Pharmacokinetic Profile (Oral Absorption)\n"
        f"Dose: {dose} mg, ka: {absorption_rate_constant:.2f} h⁻¹, t½: {half_life} h, Dosing times: {dose_times_str}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set reasonable y-axis limits
    ax.set_ylim(0, np.max(concentrations) * 1.1)
    ax.set_xlim(0, plot_duration)

    plt.tight_layout()
    return fig


def update_plot(
    dose: float,
    absorption_rate_constant: float,
    half_life: float,
    dose_times_str: str,
    plot_duration: float = 24.0,
    averaging_interval: float = 4.0,
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
        absorption_rate_constant : float
            First-order absorption rate constant (1/h)
        half_life : float
            Elimination half-life (hours)
        dose_times_str : str
            Comma-separated dosing times in 24h format (e.g., "8,19")
        plot_duration : float, optional
            Duration of simulation to plot (hours), by default 24.0
        averaging_interval : float, optional
            Time interval for computing average concentrations (hours), by default 4.0

        Returns
        -------
        fig : matplotlib.figure.Figure
            Updated plot figure
    """
    # Input validation to prevent errors
    if any(param <= 0 for param in [dose, absorption_rate_constant, half_life]):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "All numeric parameters must be positive values",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Invalid Parameters", fontsize=14)
        return fig

    # Parse dose times from string
    try:
        dose_times = [float(t.strip()) for t in dose_times_str.split(",") if t.strip()]

        # Validate dose times are within 0-24 hour range
        if not dose_times:
            raise ValueError("No dose times provided")

        for time in dose_times:
            if time < 0 or time >= 24:
                raise ValueError(f"Dose time {time} must be between 0 and 24 hours")

    except (ValueError, AttributeError) as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            f"Invalid dose times format.\nUse comma-separated hours (0-24).\nExample: '8,19' for 8am and 7pm\nError: {str(e)}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Invalid Dose Times", fontsize=14)
        return fig

    return create_pk_plot(
        dose,
        absorption_rate_constant,
        half_life,
        dose_times,
        plot_duration,
        averaging_interval,
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
            maximum=10,
            value=1.0,
            step=0.1,
            label="Absorption Rate Constant (1/h)",
            info="First-order absorption rate constant (ka)",
        ),
        gr.Slider(
            minimum=0.5,
            maximum=48,
            value=8,
            step=0.5,
            label="Half-life (hours)",
            info="Time for drug concentration to decrease by 50%",
        ),
        gr.Textbox(
            value="8,19",
            label="Dosing Times (hours)",
            info="Comma-separated times in 24h format (e.g., '8,19' for 8am and 7pm)",
            placeholder="8,19",
        ),
        gr.Slider(
            minimum=1,
            maximum=168,
            value=24,
            step=1,
            label="Plot Duration (hours)",
            info="Total duration to simulate and plot (1-168 hours, default 24h)",
        ),
        gr.Slider(
            minimum=0.5,
            maximum=24,
            value=4.0,
            step=0.5,
            label="Averaging Interval (hours)",
            info="Time interval for computing average concentrations (0.5-24 hours)",
        ),
    ]

    # Define output component
    outputs = gr.Plot(label="Pharmacokinetic Profile")

    # Create interface with professional styling and model assumptions
    with gr.Blocks(theme=gr.themes.Soft(), title="Pharmacokinetic Simulation Tool") as iface:
        gr.Markdown("# Pharmacokinetic Simulation Tool")
        gr.Markdown("""
        Simulate plasma drug concentrations over time with custom oral dosing schedules.
        Adjust the absorption rate constant (ka) and half-life parameters, and specify dosing times
        within a 24-hour period to see how drug accumulation patterns change. 
        
        <a href="https://go.drugbank.com/drugs/" target="_blank">Find drug parameters (half-life, etc.) on DrugBank</a>
        
        Example: enter "8,19" for doses at 8am and 7pm daily.
        """)
        
        with gr.Accordion("Model Assumptions & Mathematical Formula", open=False):
            gr.Markdown("""
            ### Single Compartment Pharmacokinetic Model
            
            **Key Assumptions:**
            - **Single compartment model**: The body is treated as a single, well-mixed compartment
            - **First-order absorption**: Drug absorption from the gut follows first-order kinetics
            - **First-order elimination**: Drug elimination follows first-order kinetics
            - **Volume of distribution = 1L**: For normalized concentration units
            - **Linear kinetics**: No saturation effects (valid for therapeutic doses)
            - **Superposition principle**: Multiple doses are additive in effect
            
            **Mathematical Formula:**
            
            For oral absorption with first-order kinetics:
            
            ```
            C(t) = (Dose × ka) / (ka - ke) × [exp(-ke × t) - exp(-ka × t)]
            ```
            
            Where:
            - **C(t)** = Plasma concentration at time t
            - **Dose** = Amount of drug administered (mg)
            - **ka** = Absorption rate constant (h⁻¹)
            - **ke** = Elimination rate constant (h⁻¹) = ln(2) / half-life
            - **t** = Time since dose administration (hours)
            
            **For multiple doses:** The concentration at any time is the sum of contributions from all previous doses (superposition principle).
            
            **Special case:** When ka = ke (flip-flop kinetics):
            ```
            C(t) = Dose × ka × t × exp(-ka × t)
            ```
            """)
        
        with gr.Row():
            with gr.Column():
                dose_input = inputs[0]
                ka_input = inputs[1]
                half_life_input = inputs[2]
                dose_times_input = inputs[3]
                plot_duration_input = inputs[4]
                averaging_interval_input = inputs[5]
            
            with gr.Column():
                plot_output = outputs
        
        # Connect inputs to the update function
        for input_component in [dose_input, ka_input, half_life_input, dose_times_input, plot_duration_input, averaging_interval_input]:
            input_component.change(
                fn=update_plot,
                inputs=[dose_input, ka_input, half_life_input, dose_times_input, plot_duration_input, averaging_interval_input],
                outputs=plot_output
            )
        
        # Set initial plot
        iface.load(
            fn=update_plot,
            inputs=[dose_input, ka_input, half_life_input, dose_times_input, plot_duration_input, averaging_interval_input],
            outputs=plot_output
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
