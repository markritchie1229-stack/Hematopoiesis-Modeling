#!/usr/bin/env python3
"""
Convert the true-root lineage pickle to CSV files.

The script automatically reads and writes in the same folder as this
Python script, so it can be launched from any Terminal directory.

Input:
    hematopoiesis_generations_with_root_lineage_FIXED.pkl

Outputs:
    hematopoiesis_generation_sizes.csv
    hematopoiesis_lineage_cells.csv
    hematopoiesis_lineage_edges.csv
"""

from pathlib import Path
import pickle
import pandas as pd


# =========================================================
# FILE LOCATIONS
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PICKLE_FILE = SCRIPT_DIR / "hematopoiesis_generations_with_root_lineage_FIXED.pkl"

GENERATION_CSV = SCRIPT_DIR / "hematopoiesis_generation_sizes.csv"

LINEAGE_CSV = SCRIPT_DIR / "hematopoiesis_lineage_cells.csv"

EDGES_CSV = SCRIPT_DIR / "hematopoiesis_lineage_edges.csv"


# =========================================================
# CHECK INPUT
# =========================================================

if not PICKLE_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find the pickle file:\n{PICKLE_FILE}\n\n"
        "Run the simulation script first so that the pickle is created."
    )


# =========================================================
# LOAD PICKLE
# =========================================================

print("Reading pickle:")
print(PICKLE_FILE)
print()

with open(PICKLE_FILE, "rb") as f:
    simulation_data = pickle.load(f)


# =========================================================
# GENERATION COUNTS
# =========================================================

generation_rows = []

for root_id, root_data in sorted(simulation_data.items()):

    generation_counts = root_data["generation_counts"]

    for generation in sorted(generation_counts.keys()):

        counts = generation_counts[generation]

        generation_rows.append({
            "Root_ID": root_id,
            "Generation": generation,
            "LT-HSC_Count": counts["LT-HSC"],
            "ST-HSC_Count": counts["ST-HSC"],
            "MPP_Count": counts["MPP"],
        })


generation_df = pd.DataFrame(generation_rows)

generation_df.to_csv(
    GENERATION_CSV,
    index=False,
)


# =========================================================
# INDIVIDUAL CELL LINEAGE RECORDS
# =========================================================

lineage_rows = []

for root_id, root_data in sorted(simulation_data.items()):

    lineage = root_data["lineage"]

    for unique_id, cell in sorted(
        lineage.items(),
        key=lambda item: item[0],
    ):

        lineage_rows.append({
            "Root_ID": root_id,
            "Unique_ID": cell["unique_id"],
            "Parent_ID": cell["parent_id"],
            "Cell_Type": cell["cell_type"],
            "Generation": cell["generation"],
            "Creation_Time": cell["creation_time"],
            "Division_Count": cell["division_count"],
            "Active_At_End": cell["active_at_end"],
        })


lineage_df = pd.DataFrame(lineage_rows)

lineage_df.to_csv(
    LINEAGE_CSV,
    index=False,
)


# =========================================================
# LINEAGE EDGES
# =========================================================
# This is a simplified parent -> child table that is convenient
# for network/lineage visualization in programs such as R,
# Python, Gephi, Cytoscape, or Tableau.

edge_rows = []

for root_id, root_data in sorted(simulation_data.items()):

    lineage = root_data["lineage"]

    for unique_id, cell in sorted(
        lineage.items(),
        key=lambda item: item[0],
    ):

        parent_id = cell["parent_id"]

        # Root cells have no parent and therefore do not create
        # a parent-child edge.
        if parent_id is not None:

            edge_rows.append({
                "Root_ID": root_id,
                "Parent_ID": parent_id,
                "Child_ID": unique_id,
                "Child_Type": cell["cell_type"],
                "Child_Generation": cell["generation"],
                "Creation_Time": cell["creation_time"],
            })


edges_df = pd.DataFrame(edge_rows)

edges_df.to_csv(
    EDGES_CSV,
    index=False,
)


# =========================================================
# VALIDATION
# =========================================================

root_ids = sorted(simulation_data.keys())

root_cells = lineage_df[
    lineage_df["Parent_ID"].isna()
]

# Since root IDs are actual root cells, there should be exactly
# one parentless cell for each root.
if len(root_cells) != len(root_ids):
    raise RuntimeError(
        "Validation failed: expected exactly one root cell per Root_ID."
    )

if set(root_cells["Root_ID"]) != set(root_ids):
    raise RuntimeError(
        "Validation failed: parentless cells do not match the Root_IDs."
    )

# Every individual Unique_ID must occur exactly once.
if lineage_df["Unique_ID"].duplicated().any():
    raise RuntimeError(
        "Validation failed: duplicate Unique_ID values were found."
    )


# =========================================================
# SUMMARY
# =========================================================

print("=" * 70)
print("CSV CONVERSION COMPLETE")
print("=" * 70)
print()

print("Input pickle:")
print(PICKLE_FILE)
print()

print("Generation CSV:")
print(GENERATION_CSV)
print(f"Rows: {len(generation_df):,}")
print()

print("Individual lineage CSV:")
print(LINEAGE_CSV)
print(f"Rows: {len(lineage_df):,}")
print()

print("Parent-child edge CSV:")
print(EDGES_CSV)
print(f"Rows: {len(edges_df):,}")
print()

print("Root cells:", len(root_ids))
print("Unique individual cells:", lineage_df["Unique_ID"].nunique())
print("Maximum generation:", int(lineage_df["Generation"].max()))
print()
print("Validation: PASSED")
print("=" * 70)
