#!/usr/bin/env python3
"""
Integrated Hematopoiesis Sensitivity Analysis Script (Average Perturbation)

Updates:
- P3A correctly produces 1 ST-HSC + death.
- P4A correctly produces 2 ST-HSCs + death.
- P3B correctly produces 1 MPP + death.
- P4B correctly produces 2 MPPs + death.
- ST-HSC timing matches Dr. Paula Vaquez's model:
      ST_HSC_TIME_MEAN = 12
      ST_HSC_TIME_STD  = 3.8

Sensitivity methodology:
- Sensitivity is evaluated at iterations 50, 100, and 200.
- Every perturbation uses the SAME random seed as its
  corresponding baseline simulation.
- The four perturbations for each continuous parameter are
  averaged.
- The mean absolute change at the selected iteration
  is used as the sensitivity value.
- Division-threshold parameters similarly use the mean
  final-state change among their tested perturbations.

Visualization:
- A separate ranked horizontal sensitivity bar chart is
  generated for iterations 50, 100, and 200.
- Each plot has its own ranked legend/key on the right.
- Legend ordering exactly matches the sensitivity ranking.
- Sensitivity values are shown directly on the bars.
- A separate generation-based visualization is generated
  for each analysis iteration.
- Output files are automatically saved beside this script.
"""

import numpy as np
import copy
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Output Directory
# ============================================================

# Save output files in the SAME DIRECTORY as this Python script.
# This prevents read-only-directory errors.

SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# Analysis Iterations
# ============================================================

ANALYSIS_ITERATIONS = [
    50,
    100,
    200
]


# ============================================================
# Output File Names
# ============================================================

def get_sensitivity_output(iteration):
    """
    Return the output path for a sensitivity plot.
    """

    return (
        SCRIPT_DIR
        / f"hematopoiesis_sensitivity_ranking_iteration_{iteration}.png"
    )


def get_generation_output(iteration):
    """
    Return the output path for a generation plot.
    """

    return (
        SCRIPT_DIR
        / f"hematopoiesis_generation_counts_iteration_{iteration}.png"
    )


# ============================================================
# Simulation Settings
# ============================================================

INITIAL_CELLS = 1000
RANDOM_SEED = 42


# ============================================================
# Base Parameters
# ============================================================

base_params = {
    "LT_HSC_DIVISION_THRESHOLD": 10,
    "ST_HSC_DIVISION_THRESHOLD": 3,

    "PAA": 0.0025,
    "PAB": 0.0025,

    "PQA": 0.05,
    "PQB": 0.15,

    "P_EXIT_QUIESCENCE_A": 0.05,
    "P_EXIT_QUIESCENCE_B": 0.15,

    "P1A": 0.60,
    "P2A": 0.3090,
    "P3A": 0.0147,
    "P4A": 0.0763,

    "P1B": 0.50,
    "P2B": 0.0117,
    "P3B": 0.216,
    "P4B": 0.2723,

    "LT_HSC_TIME_MEAN": 40,
    "LT_HSC_TIME_STD": 19.6,

    # Corrected to match Dr. Vaquez's model
    "ST_HSC_TIME_MEAN": 12,
    "ST_HSC_TIME_STD": 3.8,
}


# ============================================================
# Normalize Fate Probabilities
# ============================================================

def normalize_fate(params, prefix):
    """
    Normalize P1/P2/P3/P4 fate probabilities.

    prefix:
        A = LT-HSC
        B = ST-HSC
    """

    keys = [
        f"P1{prefix}",
        f"P2{prefix}",
        f"P3{prefix}",
        f"P4{prefix}"
    ]

    vals = np.array(
        [params[k] for k in keys],
        dtype=float
    )

    vals = np.clip(vals, 1e-8, None)

    vals = vals / np.sum(vals)

    for i, key in enumerate(keys):
        params[key] = vals[i]


# ============================================================
# Simulation
# ============================================================

def simulate(
    params,
    seed=0,
    steps=50,
    initial_cells=INITIAL_CELLS
):
    """
    Run the simplified hematopoiesis simulation.

    The returned counts represent the population after
    'steps' iterations.

    Returns:
        counts[(cell_type, generation)] = number of cells
    """

    # Make a copy so that normalization and other operations
    # do not modify the original parameter dictionary.
    params = copy.deepcopy(params)

    rng = np.random.default_rng(seed)

    cells = []

    # --------------------------------------------------------
    # Initial LT-HSC population
    # --------------------------------------------------------

    for _ in range(initial_cells):

        cells.append({
            "type": "LT",
            "state": "Active",
            "divisions": 0,
            "generation": 0,
            "timer": rng.normal(
                params["LT_HSC_TIME_MEAN"],
                params["LT_HSC_TIME_STD"]
            )
        })

    # --------------------------------------------------------
    # Main simulation
    # --------------------------------------------------------

    for iteration in range(steps):

        new_cells = []

        for cell in cells:

            if cell["state"] == "Dead":
                continue

            # ------------------------------------------------
            # Apoptosis
            # ------------------------------------------------

            if cell["state"] == "Active":

                if cell["type"] == "LT":

                    if rng.random() < params["PAA"]:
                        continue

                else:

                    if rng.random() < params["PAB"]:
                        continue

            # ------------------------------------------------
            # Quiescence entry
            # ------------------------------------------------

            if cell["state"] == "Active":

                if cell["type"] == "LT":

                    if rng.random() < params["PQA"]:
                        cell["state"] = "Quiescent"

                else:

                    if rng.random() < params["PQB"]:
                        cell["state"] = "Quiescent"

            # ------------------------------------------------
            # Quiescence exit
            # ------------------------------------------------

            if cell["state"] == "Quiescent":

                if cell["type"] == "LT":

                    if rng.random() < params["P_EXIT_QUIESCENCE_A"]:
                        cell["state"] = "Active"

                else:

                    if rng.random() < params["P_EXIT_QUIESCENCE_B"]:
                        cell["state"] = "Active"

            # ------------------------------------------------
            # Division threshold
            # ------------------------------------------------

            threshold = (
                params["LT_HSC_DIVISION_THRESHOLD"]
                if cell["type"] == "LT"
                else params["ST_HSC_DIVISION_THRESHOLD"]
            )

            if cell["divisions"] >= threshold:
                continue

            # ------------------------------------------------
            # Timer
            # ------------------------------------------------

            cell["timer"] -= 1

            if cell["timer"] > 0:

                new_cells.append(cell)

                continue

            # =================================================
            # LT-HSC DIVISION
            # =================================================

            if cell["type"] == "LT":

                normalize_fate(params, "A")

                r = rng.random()

                # P1A:
                # LT-HSC -> 2 LT-HSCs
                if r < params["P1A"]:

                    children = ["LT", "LT"]

                # P2A:
                # LT-HSC -> LT-HSC + ST-HSC
                elif r < (
                    params["P1A"]
                    + params["P2A"]
                ):

                    children = ["LT", "ST"]

                # P3A:
                # LT-HSC -> 1 ST-HSC + death
                elif r < (
                    params["P1A"]
                    + params["P2A"]
                    + params["P3A"]
                ):

                    children = ["ST"]

                # P4A:
                # LT-HSC -> 2 ST-HSCs + death
                else:

                    children = ["ST", "ST"]

            # =================================================
            # ST-HSC DIVISION
            # =================================================

            else:

                normalize_fate(params, "B")

                r = rng.random()

                # P1B:
                # ST-HSC -> 2 ST-HSCs
                if r < params["P1B"]:

                    children = ["ST", "ST"]

                # P2B:
                # ST-HSC -> ST-HSC + MPP
                elif r < (
                    params["P1B"]
                    + params["P2B"]
                ):

                    children = ["ST", "MPP"]

                # P3B:
                # ST-HSC -> 1 MPP + death
                elif r < (
                    params["P1B"]
                    + params["P2B"]
                    + params["P3B"]
                ):

                    children = ["MPP"]

                # P4B:
                # ST-HSC -> 2 MPPs + death
                else:

                    children = ["MPP", "MPP"]

            # ------------------------------------------------
            # Create daughter cells
            # ------------------------------------------------

            for ctype in children:

                if ctype == "LT":

                    timer = rng.normal(
                        params["LT_HSC_TIME_MEAN"],
                        params["LT_HSC_TIME_STD"]
                    )

                elif ctype == "ST":

                    timer = rng.normal(
                        params["ST_HSC_TIME_MEAN"],
                        params["ST_HSC_TIME_STD"]
                    )

                else:

                    # MPPs do not divide in this model.
                    timer = 0

                new_cells.append({
                    "type": ctype,
                    "state": "Active",
                    "divisions": cell["divisions"] + 1,
                    "generation": cell["generation"] + 1,
                    "timer": timer
                })

        cells = new_cells

    # ========================================================
    # Count cells by type and generation
    # ========================================================

    counts = defaultdict(int)

    for cell in cells:

        counts[
            (
                cell["type"],
                cell["generation"]
            )
        ] += 1

    return counts


# ============================================================
# Perturbation Logic
# ============================================================

def perturb_params(
    base,
    param_name,
    factor=None,
    delta=None
):
    """
    Create a perturbed copy of the base parameter set.
    """

    p = copy.deepcopy(base)

    # --------------------------------------------------------
    # Discrete parameters
    # --------------------------------------------------------

    if param_name in [
        "LT_HSC_DIVISION_THRESHOLD",
        "ST_HSC_DIVISION_THRESHOLD"
    ]:

        new_val = int(
            round(
                p[param_name] + delta
            )
        )

        p[param_name] = max(
            1,
            new_val
        )

    # --------------------------------------------------------
    # LT-HSC fate probabilities
    # --------------------------------------------------------

    elif param_name in [
        "P1A",
        "P2A",
        "P3A",
        "P4A"
    ]:

        p[param_name] *= factor

        normalize_fate(
            p,
            "A"
        )

    # --------------------------------------------------------
    # ST-HSC fate probabilities
    # --------------------------------------------------------

    elif param_name in [
        "P1B",
        "P2B",
        "P3B",
        "P4B"
    ]:

        p[param_name] *= factor

        normalize_fate(
            p,
            "B"
        )

    # --------------------------------------------------------
    # Other continuous parameters
    # --------------------------------------------------------

    else:

        p[param_name] *= factor

    return p


# ============================================================
# Comparison Metric
# ============================================================

def compare_counts(
    base_counts,
    pert_counts
):
    """
    Calculate total absolute difference between baseline
    and perturbed cell counts.

    Counts are grouped by cell type and generation.

    The comparison is made ONLY at the requested final
    iteration.
    """

    all_keys = (
        set(base_counts)
        |
        set(pert_counts)
    )

    return sum(
        abs(
            base_counts.get(k, 0)
            -
            pert_counts.get(k, 0)
        )
        for k in all_keys
    )


# ============================================================
# Sensitivity Analysis
# ============================================================

def run_sensitivity(steps):
    """
    Run sensitivity analysis for a specific simulation length.

    Sensitivity is calculated using the population at the
    specified final iteration.

    The mean absolute change among the tested perturbations
    is used as the sensitivity value.
    """

    print(
        f"\nRunning baseline simulation for "
        f"{steps} iterations..."
    )

    # --------------------------------------------------------
    # Baseline simulation
    # --------------------------------------------------------

    base_counts = simulate(
        base_params,
        seed=RANDOM_SEED,
        steps=steps,
        initial_cells=INITIAL_CELLS
    )

    results = {}
    perturbation_results = {}

    # --------------------------------------------------------
    # Analyze every parameter
    # --------------------------------------------------------

    for param in base_params.keys():

        diffs = []

        # ----------------------------------------------------
        # Division threshold parameters
        # ----------------------------------------------------

        if param == "LT_HSC_DIVISION_THRESHOLD":

            deltas = [-2, -1, 1, 2]

        elif param == "ST_HSC_DIVISION_THRESHOLD":

            deltas = [-1, 1]

        else:

            deltas = None

        # ----------------------------------------------------
        # Discrete parameters
        # ----------------------------------------------------

        if deltas is not None:

            for d in deltas:

                p = perturb_params(
                    base_params,
                    param,
                    delta=d
                )

                pert_counts = simulate(
                    p,
                    seed=RANDOM_SEED,
                    steps=steps,
                    initial_cells=INITIAL_CELLS
                )

                diff = compare_counts(
                    base_counts,
                    pert_counts
                )

                diffs.append(diff)

        # ----------------------------------------------------
        # Continuous parameters
        # ----------------------------------------------------

        else:

            factors = [
                0.8,
                0.9,
                1.1,
                1.2
            ]

            for f in factors:

                p = perturb_params(
                    base_params,
                    param,
                    factor=f
                )

                pert_counts = simulate(
                    p,
                    seed=RANDOM_SEED,
                    steps=steps,
                    initial_cells=INITIAL_CELLS
                )

                diff = compare_counts(
                    base_counts,
                    pert_counts
                )

                diffs.append(diff)

        # ----------------------------------------------------
        # Mean final-state sensitivity
        # ----------------------------------------------------

        results[param] = np.mean(diffs)

        perturbation_results[param] = diffs

    return (
        results,
        base_counts,
        perturbation_results
    )


# ============================================================
# Ranked Results
# ============================================================

def rank_results(results):

    return sorted(
        results.items(),
        key=lambda x: -x[1]
    )


# ============================================================
# Print Ranked Results
# ============================================================

def print_results(
    results,
    perturbation_results,
    steps
):

    ranked = rank_results(results)

    print("\n")
    print("=" * 80)

    print(
        "HEMATOPOIESIS PARAMETER SENSITIVITY RANKING"
    )

    print("=" * 80)

    print(
        f"Final simulation iteration: {steps}"
    )

    print(
        "Sensitivity metric: mean absolute change "
        "at final iteration"
    )

    print("=" * 80)

    for rank, (param, value) in enumerate(
        ranked,
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{param:<35} "
            f"{value:>12,.4f}"
        )

    print("=" * 80)

    print("\nFINAL-ITERATION PERTURBATION DETAILS")
    print("-" * 80)

    for param, diffs in perturbation_results.items():

        print(
            f"{param:<35}: "
            + ", ".join(
                f"{x:,.1f}"
                for x in diffs
            )
        )

    print("-" * 80)


# ============================================================
# Sensitivity Plot
# ============================================================

def plot_sensitivity(
    results,
    steps
):
    """
    Create a ranked horizontal sensitivity bar chart for
    the specified iteration using mean perturbation change.

    The ranked legend uses EXACTLY the same ordering as
    the sensitivity ranking.
    """

    ranked = rank_results(results)

    # Reverse order for horizontal bar chart
    ranked_plot = ranked[::-1]

    parameters = [
        x[0]
        for x in ranked_plot
    ]

    values = [
        x[1]
        for x in ranked_plot
    ]

    n = len(parameters)

    # --------------------------------------------------------
    # Highly differentiated color palette
    # --------------------------------------------------------

    colors = plt.cm.turbo(
        np.linspace(
            0.05,
            0.95,
            n
        )
    )

    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    bars = ax.barh(
        parameters,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.90
    )

    # --------------------------------------------------------
    # Values at end of bars
    # --------------------------------------------------------

    max_value = max(values)

    # Prevent problems if all values are zero.
    if max_value == 0:
        label_offset = 0.1
    else:
        label_offset = max_value * 0.012

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_width() + label_offset,

            bar.get_y()
            + bar.get_height() / 2,

            f"{value:,.1f}",

            va="center",
            ha="left",
            fontsize=8.5
        )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Maximum absolute change in final "
        "generation-based cell counts",
        fontsize=12
    )

    ax.set_ylabel(
        "Model parameter",
        fontsize=12
    )

    ax.set_title(
        "Hematopoiesis Parameter Sensitivity Analysis\n"
        f"Final-state sensitivity at iteration {steps}",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Ranked legend
    #
    # IMPORTANT:
    # The legend uses the EXACT SAME ordering as the
    # sensitivity ranking.
    # --------------------------------------------------------

    legend_handles = []

    color_lookup = {
        param: color
        for param, color in zip(
            parameters,
            colors
        )
    }

    for rank, (param, value) in enumerate(
        ranked,
        start=1
    ):

        handle = Line2D(
            [0],
            [0],

            marker="s",
            linestyle="None",

            markerfacecolor=color_lookup[param],
            markeredgecolor="black",

            markersize=9,

            label=f"{rank}. {param}"
        )

        legend_handles.append(handle)

    ax.legend(
        handles=legend_handles,

        title=(
            f"Ranked parameter key\n"
            f"(final iteration = {steps})"
        ),

        loc="upper left",

        bbox_to_anchor=(1.02, 1.0),

        fontsize=8.5,
        title_fontsize=10,

        frameon=True,

        borderaxespad=0
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout(
        rect=[
            0,
            0,
            0.74,
            1
        ]
    )

    # --------------------------------------------------------
    # Save figure beside script
    # --------------------------------------------------------

    output_file = get_sensitivity_output(
        steps
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\nSensitivity plot saved to:"
    )

    print(
        output_file
    )

    plt.show()

    plt.close(fig)


# ============================================================
# Generation-Based Visualization
# ============================================================

def plot_generation_counts(
    base_counts,
    steps
):
    """
    Plot baseline final counts by generation for the
    specified iteration.
    """

    # --------------------------------------------------------
    # Organize baseline counts
    # --------------------------------------------------------

    generations = sorted(
        set(
            generation
            for (
                cell_type,
                generation
            ) in base_counts.keys()
        )
    )

    cell_types = [
        "LT",
        "ST",
        "MPP"
    ]

    colors = {
        "LT": "#0072B2",
        "ST": "#D55E00",
        "MPP": "#009E73"
    }

    linestyles = {
        "LT": "-",
        "ST": "--",
        "MPP": "-."
    }

    markers = {
        "LT": "o",
        "ST": "s",
        "MPP": "^"
    }

    labels = {
        "LT": "LT-HSC",
        "ST": "ST-HSC",
        "MPP": "MPP"
    }

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for cell_type in cell_types:

        values = [
            base_counts.get(
                (
                    cell_type,
                    generation
                ),
                0
            )
            for generation in generations
        ]

        ax.plot(
            generations,
            values,

            color=colors[cell_type],

            linestyle=linestyles[cell_type],

            marker=markers[cell_type],

            linewidth=2.8,

            markersize=7,

            markerfacecolor="white",

            markeredgewidth=1.5,

            label=labels[cell_type]
        )

    ax.set_xlabel(
        "Generation",
        fontsize=12
    )

    ax.set_ylabel(
        "Cell count",
        fontsize=12
    )

    ax.set_title(
        "Baseline Cell Counts by Generation\n"
        f"Final population after {steps} iterations",
        fontsize=15,
        fontweight="bold"
    )

    ax.grid(
        True,
        linestyle=":",
        alpha=0.45
    )

    ax.legend(
        title="Cell lineage",
        fontsize=10,
        title_fontsize=11
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save beside script
    # --------------------------------------------------------

    output_file = get_generation_output(
        steps
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\nGeneration plot saved to:"
    )

    print(
        output_file
    )

    plt.show()

    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("HEMATOPOIESIS SENSITIVITY ANALYSIS")
    print("=" * 80)

    print(
        "Analysis iterations:"
    )

    print(
        ", ".join(
            str(x)
            for x in ANALYSIS_ITERATIONS
        )
    )

    print("=" * 80)

    # ========================================================
    # Store results for all requested iterations
    # ========================================================

    all_results = {}

    all_baseline_counts = {}

    all_perturbation_results = {}

    # ========================================================
    # Run analysis for iterations 50, 100, and 200
    # ========================================================

    for iteration in ANALYSIS_ITERATIONS:

        print("\n")
        print("#" * 80)

        print(
            f"STARTING ANALYSIS FOR ITERATION {iteration}"
        )

        print("#" * 80)

        (
            results,
            baseline_counts,
            perturbation_results
        ) = run_sensitivity(
            steps=iteration
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        all_results[iteration] = results

        all_baseline_counts[iteration] = (
            baseline_counts
        )

        all_perturbation_results[iteration] = (
            perturbation_results
        )

        # ----------------------------------------------------
        # Print ranking
        # ----------------------------------------------------

        print_results(
            results,
            perturbation_results,
            iteration
        )

        # ----------------------------------------------------
        # Create sensitivity plot
        # ----------------------------------------------------

        plot_sensitivity(
            results,
            iteration
        )

        # ----------------------------------------------------
        # Create generation plot
        # ----------------------------------------------------

        plot_generation_counts(
            baseline_counts,
            iteration
        )

    # ========================================================
    # Final Summary
    # ========================================================

    print("\n")
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        "\nSensitivity rankings were generated for:"
    )

    for iteration in ANALYSIS_ITERATIONS:

        print(
            f"  - Iteration {iteration}"
        )

    print("\nOutput files:")

    for iteration in ANALYSIS_ITERATIONS:

        print(
            f"\nIteration {iteration}:"
        )

        print(
            f"  Sensitivity:"
        )

        print(
            f"    {get_sensitivity_output(iteration)}"
        )

        print(
            f"  Generation:"
        )

        print(
            f"    {get_generation_output(iteration)}"
        )

    print("\nAll output files were saved in:")
    print(SCRIPT_DIR)
    print("=" * 80)
