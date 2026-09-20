#!/usr/bin/env python3
"""
Plot the 100 root-cell lineage trees from the generation-size CSV.

Input
-----
hematopoiesis_generation_sizes.csv

Expected columns
----------------
Root_ID
Generation
LT-HSC_Count
ST-HSC_Count
MPP_Count

What the plot shows
-------------------
There are three panels:
    1. LT-HSC count by generation
    2. ST-HSC count by generation
    3. MPP count by generation

Each line represents ONE root-cell lineage (Root_ID 0-99).

This means:
    - x-axis = generation
    - y-axis = number of cells of that type
    - each colored/gray line = one of the 100 root lineages

The script does NOT combine the 100 roots into one average line.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# FILE LOCATIONS
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIR / "hematopoiesis_generation_sizes.csv"

OUTPUT_FILE = SCRIPT_DIR / "hematopoiesis_lineage_trees_by_cell_type.png"


# =========================================================
# CHECK INPUT FILE
# =========================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_FILE}\n\n"
        "Make sure hematopoiesis_generation_sizes.csv is in the "
        "same folder as this script."
    )


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)


required_columns = {
    "Root_ID",
    "Generation",
    "LT-HSC_Count",
    "ST-HSC_Count",
    "MPP_Count",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        "The CSV is missing these required columns: "
        + ", ".join(sorted(missing))
    )


# =========================================================
# BASIC VALIDATION
# =========================================================

root_ids = sorted(df["Root_ID"].unique())

if len(root_ids) != 100:
    print(
        f"WARNING: expected 100 root lineages, "
        f"but found {len(root_ids)}."
    )

print(f"Found {len(root_ids)} root lineages.")

# Sort data so every root/generation is in the correct order.
df = df.sort_values(
    ["Root_ID", "Generation"]
).copy()


# =========================================================
# PLOT
# =========================================================

cell_types = [
    ("LT-HSC_Count", "LT-HSC"),
    ("ST-HSC_Count", "ST-HSC"),
    ("MPP_Count", "MPP"),
]

fig, axes = plt.subplots(
    3,
    1,
    figsize=(13, 16),
    sharex=True,
)


# Use a light/semitransparent line for each root so that
# overlapping trees remain visible.
for ax, (count_column, cell_label) in zip(axes, cell_types):

    for root_id in root_ids:

        root_data = df[df["Root_ID"] == root_id]

        ax.plot(
            root_data["Generation"],
            root_data[count_column],
            linewidth=1.0,
            alpha=0.30,
        )

    ax.set_ylabel("Cell count", fontsize=11)
    ax.set_title(
        f"{cell_label}: Individual Root-Cell Lineages",
        fontsize=13,
        fontweight="bold",
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.7,
        alpha=0.4,
    )


# X-axis only on the bottom panel.
axes[-1].set_xlabel(
    "Generation",
    fontsize=12,
)


fig.suptitle(
    "Hematopoiesis Lineage Trees by Generation\n"
    "Each line represents one Root_ID",
    fontsize=16,
    fontweight="bold",
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)


# =========================================================
# SAVE
# =========================================================

fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

plt.close(fig)


print()
print("Plot saved to:")
print(OUTPUT_FILE)
print()
print("Each line represents one root-cell lineage.")
print("Number of root lineages:", len(root_ids))
