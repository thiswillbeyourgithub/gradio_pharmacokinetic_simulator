# Pharmacokinetic Simulation Tool

A Gradio-based web interface for simulating plasma drug concentrations over time with multiple dosing regimens. This interactive tool helps visualize drug accumulation patterns and pharmacokinetic behavior based on key parameters like absorption rate, elimination half-life, and dosing schedules.

## Background

I created this tool because I wanted to work my intuition on pharmacokinetic concepts - specifically how different dosing schedules, absorption rates, and elimination half-lives affect drug concentration profiles over time. Having an interactive visualization makes it much easier to understand these relationships compared to static equations or tables.

## Features

- **Interactive Web Interface**: Easy-to-use Gradio interface with real-time plot updates
- **Flexible Dosing Schedules**: Support for custom dosing times within 24-hour periods (e.g., "8,19" for 8am and 7pm)
- **Multiple Simulation Duration**: Plot from 1 to 30 days to see steady-state behavior
- **Rolling Average Calculations**: Smooth concentration curves with configurable averaging intervals
- **Professional Visualizations**: High-quality matplotlib plots with annotations for Cmax and Tmax
- **Mathematical Model Details**: Built-in explanations of the underlying pharmacokinetic equations
- **Input Validation**: Robust error handling for invalid parameters
- **Drug Parameter Lookup**: Direct links to DrugBank for finding real drug parameters

## Pharmacokinetic Model

The simulation uses a **single-compartment model** with:
- First-order absorption from the gut
- First-order elimination
- Superposition principle for multiple doses

**Mathematical Formula:**
```
C(t) = (Dose × ka) / (ka - ke) × [exp(-ke × t) - exp(-ka × t)]
```

Where:
- **C(t)** = Plasma concentration at time t
- **ka** = Absorption rate constant (h⁻¹)
- **ke** = Elimination rate constant = ln(2) / half-life
- **t** = Time since dose administration

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python script.py
```

Then open your web browser and navigate to `http://127.0.0.1:7860` to access the interface.

### Interface Controls

- **Dose (mg)**: Amount of drug per dose (1-1000 mg)
- **Absorption Rate Constant**: First-order absorption rate (0.1-10 h⁻¹)
- **Half-life**: Drug elimination half-life (0.5-48 hours)
- **Dosing Times**: Comma-separated times in 24h format (e.g., "8,19" for twice daily)
- **Plot Duration**: Total simulation time (1-30 days)
- **Averaging Interval**: Time window for rolling averages (0.5-24 hours)

### Example Use Cases

1. **Twice Daily Dosing**: Set dosing times to "8,19" to simulate morning and evening doses
2. **Once Daily**: Use "8" for a single daily dose at 8am
3. **Three Times Daily**: Try "8,14,20" for dosing every ~6-8 hours
4. **Steady State Analysis**: Extend plot duration to 7-14 days to see accumulation patterns

## Educational Value

This tool is particularly useful for:
- Understanding drug accumulation with repeated dosing
- Visualizing the impact of dosing frequency on peak and trough levels
- Comparing different absorption rates and elimination half-lives
- Learning about steady-state pharmacokinetics
- Exploring the relationship between Cmax, Tmax, and dosing parameters

## Technical Details

- **Framework**: Gradio for web interface
- **Plotting**: Matplotlib with professional styling
- **Calculations**: NumPy for efficient numerical computations
- **Validation**: Comprehensive input validation and error handling
- **Backend**: Non-interactive matplotlib backend for web deployment

## Acknowledgments

Created with assistance from [aider.chat](https://github.com/Aider-AI/aider/).

Drug parameter information can be found on [DrugBank](https://go.drugbank.com/drugs/).

