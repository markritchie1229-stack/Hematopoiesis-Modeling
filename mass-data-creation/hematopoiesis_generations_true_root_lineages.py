#!/usr/bin/env python3
"""
Hematopoiesis Generation + True Root Lineage Simulation

Runs 100 independent simulations. Each simulation begins with exactly
one LT-HSC root cell.

Root IDs:
    0 through 99

Every individual cell has:
    unique_id
    parent_id
    root_id
    cell_type
    generation
    creation_time
    division_count
    active_at_end

The simulation records:
    1. generation_counts: counts by generation and cell type
    2. lineage: individual-cell records for reconstructing each tree

Important:
- Root cells are actual individual cells, not whole-population replicates.
- Each root cell has unique_id == root_id and parent_id == None.
- Every daughter gets a globally unique ID and the direct parent's ID.
- MPPs are terminal and do not divide.
- LT-HSC-generated ST-HSC daughters are correctly carried into the
  ST-HSC population so they can later produce MPPs.
- No external packages are required beyond NumPy.
"""

from pathlib import Path
import numpy as np
import pickle
import time


# =========================================================
# FILE LOCATIONS
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PICKLE_FILE = SCRIPT_DIR / "hematopoiesis_generations_with_root_lineage_FIXED.pkl"


# =========================================================
# PARAMETERS
# =========================================================

params = {
    "LT_HSC_DIVISION_THRESHOLD": 10,
    "ST_HSC_DIVISION_THRESHOLD": 3,

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
    "ST_HSC_TIME_MEAN": 12,
    "ST_HSC_TIME_STD": 3.8,
}


# =========================================================
# GLOBAL UNIQUE CELL-ID COUNTER
# =========================================================

# Root IDs 0-99 are reserved for the 100 root cells.
next_cell_id = 100


def get_new_cell_id():
    """Return the next globally unique descendant cell ID."""
    global next_cell_id

    cell_id = next_cell_id
    next_cell_id += 1

    return cell_id


# =========================================================
# CELL CREATION
# =========================================================

def create_cell(
    unique_id,
    parent_id,
    root_id,
    cell_type,
    generation,
    timer,
    creation_time,
):
    """Create one individual cell record."""

    return {
        "unique_id": unique_id,
        "parent_id": parent_id,
        "root_id": root_id,
        "cell_type": cell_type,
        "generation": generation,
        "timer": timer,
        "creation_time": creation_time,
        "division_count": 0,
    }


# =========================================================
# REGISTER CELL
# =========================================================

def register_cell(cell, lineage, generation_counts):
    """
    Add one cell to the full lineage data and generation summary.
    """

    cell_id = cell["unique_id"]
    generation = cell["generation"]
    cell_type = cell["cell_type"]

    lineage[cell_id] = {
        "unique_id": cell_id,
        "parent_id": cell["parent_id"],
        "root_id": cell["root_id"],
        "cell_type": cell_type,
        "generation": generation,
        "creation_time": cell["creation_time"],
        "division_count": cell["division_count"],
        "active_at_end": False,
    }

    generation_counts.setdefault(
        generation,
        {
            "LT-HSC": 0,
            "ST-HSC": 0,
            "MPP": 0,
        },
    )

    generation_counts[generation][cell_type] += 1


# =========================================================
# SIMULATE ONE ROOT
# =========================================================

def simulate_root(root_id, steps, params, seed):
    """
    Simulate one lineage beginning from one root LT-HSC.

    root_id is the actual unique_id of the root cell.
    """

    # Separate reproducible random streams for LT- and ST-HSC decisions.
    lt_rng = np.random.RandomState(seed + root_id)
    st_rng = np.random.RandomState(seed + 10000 + root_id)

    # -----------------------------------------------------
    # Create the single root LT-HSC
    # -----------------------------------------------------

    root_timer = max(
        2,
        int(
            lt_rng.normal(
                params["LT_HSC_TIME_MEAN"],
                params["LT_HSC_TIME_STD"],
            )
        ),
    )

    root = create_cell(
        unique_id=root_id,
        parent_id=None,
        root_id=root_id,
        cell_type="LT-HSC",
        generation=0,
        timer=root_timer,
        creation_time=0,
    )

    # Active LT-HSCs and ST-HSCs are stored separately because
    # they use different division rules and timing distributions.
    active_lt = [root]
    active_st = []

    # MPPs are terminal in this model.
    all_mpp = []

    # Complete cell-level lineage record.
    lineage = {}

    # Generation-based summary.
    generation_counts = {}

    register_cell(
        root,
        lineage,
        generation_counts,
    )

    # -----------------------------------------------------
    # Main simulation loop
    # -----------------------------------------------------

    for current_time in range(steps):

        next_lt = []
        next_st = []
        new_mpp = []

        # =================================================
        # LT-HSCs
        # =================================================

        for cell in active_lt:

            cell["timer"] -= 1

            if cell["timer"] > 0:
                next_lt.append(cell)
                continue

            if cell["division_count"] >= params["LT_HSC_DIVISION_THRESHOLD"]:
                continue

            fate = lt_rng.rand()
            new_generation = cell["generation"] + 1

            # -------------------------------------------------
            # P1A: LT-HSC -> 2 LT-HSC
            # -------------------------------------------------

            if fate < params["P1A"]:

                for _ in range(2):

                    child_id = get_new_cell_id()

                    child_timer = max(
                        2,
                        int(
                            lt_rng.normal(
                                params["LT_HSC_TIME_MEAN"],
                                params["LT_HSC_TIME_STD"],
                            )
                        ),
                    )

                    child = create_cell(
                        unique_id=child_id,
                        parent_id=cell["unique_id"],
                        root_id=root_id,
                        cell_type="LT-HSC",
                        generation=new_generation,
                        timer=child_timer,
                        creation_time=current_time,
                    )

                    next_lt.append(child)
                    register_cell(
                        child,
                        lineage,
                        generation_counts,
                    )

            # -------------------------------------------------
            # P2A: LT-HSC -> 1 LT-HSC + 1 ST-HSC
            # -------------------------------------------------

            elif fate < params["P1A"] + params["P2A"]:

                # LT daughter
                child_id = get_new_cell_id()

                child_timer = max(
                    2,
                    int(
                        lt_rng.normal(
                            params["LT_HSC_TIME_MEAN"],
                            params["LT_HSC_TIME_STD"],
                        )
                    ),
                )

                child = create_cell(
                    unique_id=child_id,
                    parent_id=cell["unique_id"],
                    root_id=root_id,
                    cell_type="LT-HSC",
                    generation=new_generation,
                    timer=child_timer,
                    creation_time=current_time,
                )

                next_lt.append(child)
                register_cell(
                    child,
                    lineage,
                    generation_counts,
                )

                # ST daughter
                child_id = get_new_cell_id()

                child_timer = max(
                    2,
                    int(
                        st_rng.normal(
                            params["ST_HSC_TIME_MEAN"],
                            params["ST_HSC_TIME_STD"],
                        )
                    ),
                )

                child = create_cell(
                    unique_id=child_id,
                    parent_id=cell["unique_id"],
                    root_id=root_id,
                    cell_type="ST-HSC",
                    generation=new_generation,
                    timer=child_timer,
                    creation_time=current_time,
                )

                # FIXED: this ST-HSC is carried forward and can divide.
                next_st.append(child)
                register_cell(
                    child,
                    lineage,
                    generation_counts,
                )

            # -------------------------------------------------
            # P3A: LT-HSC -> 1 ST-HSC + death
            # -------------------------------------------------

            elif fate < (
                params["P1A"]
                + params["P2A"]
                + params["P3A"]
            ):

                child_id = get_new_cell_id()

                child_timer = max(
                    2,
                    int(
                        st_rng.normal(
                            params["ST_HSC_TIME_MEAN"],
                            params["ST_HSC_TIME_STD"],
                        )
                    ),
                )

                child = create_cell(
                    unique_id=child_id,
                    parent_id=cell["unique_id"],
                    root_id=root_id,
                    cell_type="ST-HSC",
                    generation=new_generation,
                    timer=child_timer,
                    creation_time=current_time,
                )

                # FIXED: this ST-HSC continues into the ST compartment.
                next_st.append(child)
                register_cell(
                    child,
                    lineage,
                    generation_counts,
                )

            # -------------------------------------------------
            # P4A: LT-HSC -> 2 ST-HSC + death
            # -------------------------------------------------

            else:

                for _ in range(2):

                    child_id = get_new_cell_id()

                    child_timer = max(
                        2,
                        int(
                            st_rng.normal(
                                params["ST_HSC_TIME_MEAN"],
                                params["ST_HSC_TIME_STD"],
                            )
                        ),
                    )

                    child = create_cell(
                        unique_id=child_id,
                        parent_id=cell["unique_id"],
                        root_id=root_id,
                        cell_type="ST-HSC",
                        generation=new_generation,
                        timer=child_timer,
                        creation_time=current_time,
                    )

                    # FIXED: both ST daughters continue and can divide.
                    next_st.append(child)
                    register_cell(
                        child,
                        lineage,
                        generation_counts,
                    )

            cell["division_count"] += 1
            lineage[cell["unique_id"]]["division_count"] = cell["division_count"]

        # =================================================
        # ST-HSCs
        # =================================================

        # IMPORTANT:
        # next_st already contains ST-HSC daughters produced above.
        # We also need to process the previously active ST-HSCs here.
        # Newly produced ST daughters are not divided again in the same
        # time step; they become available during the next time step.

        st_survivors = list(next_st)

        for cell in active_st:

            cell["timer"] -= 1

            if cell["timer"] > 0:
                st_survivors.append(cell)
                continue

            if cell["division_count"] >= params["ST_HSC_DIVISION_THRESHOLD"]:
                continue

            fate = st_rng.rand()
            new_generation = cell["generation"] + 1

            # -------------------------------------------------
            # P1B: ST-HSC -> 2 ST-HSC
            # -------------------------------------------------

            if fate < params["P1B"]:

                for _ in range(2):

                    child_id = get_new_cell_id()

                    child_timer = max(
                        2,
                        int(
                            st_rng.normal(
                                params["ST_HSC_TIME_MEAN"],
                                params["ST_HSC_TIME_STD"],
                            )
                        ),
                    )

                    child = create_cell(
                        unique_id=child_id,
                        parent_id=cell["unique_id"],
                        root_id=root_id,
                        cell_type="ST-HSC",
                        generation=new_generation,
                        timer=child_timer,
                        creation_time=current_time,
                    )

                    st_survivors.append(child)
                    register_cell(
                        child,
                        lineage,
                        generation_counts,
                    )

            # -------------------------------------------------
            # P2B: ST-HSC -> 1 ST-HSC + 1 MPP
            # -------------------------------------------------

            elif fate < params["P1B"] + params["P2B"]:

                # ST daughter
                child_id = get_new_cell_id()

                child_timer = max(
                    2,
                    int(
                        st_rng.normal(
                            params["ST_HSC_TIME_MEAN"],
                            params["ST_HSC_TIME_STD"],
                        )
                    ),
                )

                child = create_cell(
                    unique_id=child_id,
                    parent_id=cell["unique_id"],
                    root_id=root_id,
                    cell_type="ST-HSC",
                    generation=new_generation,
                    timer=child_timer,
                    creation_time=current_time,
                )

                st_survivors.append(child)
                register_cell(
                    child,
                    lineage,
                    generation_counts,
                )

                # MPP daughter
                child_id = get_new_cell_id()

                child = create_cell(
                    unique_id=child_id,
                    parent_id=cell["unique_id"],
                    root_id=root_id,
                    cell_type="MPP",
                    generation=new_generation,
                    timer=0,
                    creation_time=current_time,
                )

                new_mpp.append(child)
                register_cell(
                    child,
                    lineage,
                    generation_counts,
                )

            # -------------------------------------------------
            # P3B: ST-HSC -> 1 MPP + death
            # -------------------------------------------------

            elif fate < (
                params["P1B"]
                + params["P2B"]
                + params["P3B"]
            ):

                child_id = get_new_cell_id()

                child = create_cell(
                    unique_id=child_id,
                    parent_id=cell["unique_id"],
                    root_id=root_id,
                    cell_type="MPP",
                    generation=new_generation,
                    timer=0,
                    creation_time=current_time,
                )

                new_mpp.append(child)
                register_cell(
                    child,
                    lineage,
                    generation_counts,
                )

            # -------------------------------------------------
            # P4B: ST-HSC -> 2 MPP + death
            # -------------------------------------------------

            else:

                for _ in range(2):

                    child_id = get_new_cell_id()

                    child = create_cell(
                        unique_id=child_id,
                        parent_id=cell["unique_id"],
                        root_id=root_id,
                        cell_type="MPP",
                        generation=new_generation,
                        timer=0,
                        creation_time=current_time,
                    )

                    new_mpp.append(child)
                    register_cell(
                        child,
                        lineage,
                        generation_counts,
                    )

            cell["division_count"] += 1
            lineage[cell["unique_id"]]["division_count"] = cell["division_count"]

        # -----------------------------------------------------
        # Advance to next time step
        # -----------------------------------------------------

        active_lt = next_lt
        active_st = st_survivors
        all_mpp.extend(new_mpp)

    # -----------------------------------------------------
    # Mark cells present at simulation end
    # -----------------------------------------------------

    active_ids = {
        cell["unique_id"]
        for cell in active_lt + active_st + all_mpp
    }

    for cell_id in lineage:
        lineage[cell_id]["active_at_end"] = cell_id in active_ids

    return {
        "generation_counts": generation_counts,
        "lineage": lineage,
    }


# =========================================================
# MULTI-ROOT DRIVER
# =========================================================

def run_simulation(n_roots, steps, params, seed=42):
    """
    Run n_roots true root-cell lineage simulations.

    Each root:
        unique_id = root_id
        parent_id = None
    """

    global next_cell_id
    next_cell_id = n_roots

    print("\nStarted:", time.ctime())
    print("Output folder:", SCRIPT_DIR)
    print()

    results = {}

    for root_id in range(n_roots):

        print(
            f"Running root {root_id + 1} of {n_roots} "
            f"(Root_ID = {root_id})..."
        )

        results[root_id] = simulate_root(
            root_id=root_id,
            steps=steps,
            params=params,
            seed=seed,
        )

    print("\nFinished:", time.ctime())
    print()

    return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    ROOTS = 100

    # One true starting LT-HSC per root.
    STEPS = params["LT_HSC_DIVISION_THRESHOLD"] * int(
        params["LT_HSC_TIME_MEAN"] + 10
    )

    print("=" * 70)
    print("HEMATOPOIESIS ROOT-LINEAGE SIMULATION")
    print("=" * 70)
    print()
    print("Number of root cells:", ROOTS)
    print("Root IDs:", f"0 through {ROOTS - 1}")
    print("Simulation steps:", STEPS)
    print()

    data = run_simulation(
        n_roots=ROOTS,
        steps=STEPS,
        params=params,
        seed=42,
    )

    print()
    print("Saving results...")

    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(data, f)

    print()
    print("Saved pickle:")
    print(PICKLE_FILE)
    print()
    print("Saved data for", len(data), "root cells.")
    print("Root IDs:", list(data.keys()))
    print("=" * 70)
