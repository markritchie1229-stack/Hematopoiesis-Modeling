# These set the parameters that the model will run off of and ensure that the parameters will be used to model the cell division of 100 unique seeds. Creates your pickle file
import numpy as np
import pickle
import time
from tqdm import tqdm

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
    "ST_HSC_TIME_STD": 3.8
}

# =========================================================
# SIMULATE ONE ROOT CELL — parents counted once
# =========================================================
def simulate_root(root_id, lt_start, st_start, steps, params, seed):

    lt_rng = np.random.RandomState(seed + root_id)
    st_rng = np.random.RandomState(seed + 10000 + root_id)

    generation_counts = {}
    # Initialize gen 0
    generation_counts[0] = {"LT-HSC": lt_start, "ST-HSC": st_start, "MPP": 0}

    # Initialize individual cells
    lt_cells = [{"gen": 0,
                 "timer": max(2, int(lt_rng.normal(params["LT_HSC_TIME_MEAN"],
                                                   params["LT_HSC_TIME_STD"])))}
                for _ in range(lt_start)]

    st_cells = [{"gen": 0,
                 "timer": max(2, int(st_rng.normal(params["ST_HSC_TIME_MEAN"],
                                                   params["ST_HSC_TIME_STD"])))}
                for _ in range(st_start)]

    for _ in range(steps):
        new_lt = []
        new_st = []
        new_mpp = {}

        # ---------- LT CELLS ----------
        for cell in lt_cells:
            cell["timer"] -= 1
            if cell["timer"] <= 0:
                fate = lt_rng.rand()
                new_gen = cell["gen"] + 1

                # symmetric self-renew → 2 LT
                if fate < params["P1A"]:
                    for _ in range(2):
                        new_lt.append({
                            "gen": new_gen,
                            "timer": max(2, int(lt_rng.normal(
                                params["LT_HSC_TIME_MEAN"],
                                params["LT_HSC_TIME_STD"])))
                        })
                    # Count new generation
                    generation_counts.setdefault(new_gen, {"LT-HSC": 0, "ST-HSC": 0, "MPP": 0})
                    generation_counts[new_gen]["LT-HSC"] += 2

                # asymmetric → 1 LT + 1 ST
                elif fate < params["P1A"] + params["P2A"]:
                    new_lt.append({
                        "gen": new_gen,
                        "timer": max(2, int(lt_rng.normal(
                            params["LT_HSC_TIME_MEAN"],
                            params["LT_HSC_TIME_STD"])))
                    })
                    new_st.append({
                        "gen": new_gen,
                        "timer": max(2, int(st_rng.normal(
                            params["ST_HSC_TIME_MEAN"],
                            params["ST_HSC_TIME_STD"])))
                    })
                    generation_counts.setdefault(new_gen, {"LT-HSC": 0, "ST-HSC": 0, "MPP": 0})
                    generation_counts[new_gen]["LT-HSC"] += 1
                    generation_counts[new_gen]["ST-HSC"] += 1

                # symmetric diff → 1 ST
                elif fate < params["P1A"] + params["P2A"] + params["P3A"]:
                    new_st.append({
                        "gen": new_gen,
                        "timer": max(2, int(st_rng.normal(
                            params["ST_HSC_TIME_MEAN"],
                            params["ST_HSC_TIME_STD"])))
                    })
                    generation_counts.setdefault(new_gen, {"LT-HSC": 0, "ST-HSC": 0, "MPP": 0})
                    generation_counts[new_gen]["ST-HSC"] += 1

                # full diff → 2 ST
                else:
                    for _ in range(2):
                        new_st.append({
                            "gen": new_gen,
                            "timer": max(2, int(st_rng.normal(
                                params["ST_HSC_TIME_MEAN"],
                                params["ST_HSC_TIME_STD"])))
                        })
                    generation_counts.setdefault(new_gen, {"LT-HSC": 0, "ST-HSC": 0, "MPP": 0})
                    generation_counts[new_gen]["ST-HSC"] += 2
            else:
                # Still alive, stays in its generation — already counted
                new_lt.append(cell)

        # ---------- ST CELLS ----------
        st_next = []
        for cell in st_cells:
            cell["timer"] -= 1
            if cell["timer"] <= 0:
                fate = st_rng.rand()
                new_gen = cell["gen"] + 1

                # symmetric self-renew → 2 ST
                if fate < params["P1B"]:
                    for _ in range(2):
                        st_next.append({
                            "gen": new_gen,
                            "timer": max(2, int(st_rng.normal(
                                params["ST_HSC_TIME_MEAN"],
                                params["ST_HSC_TIME_STD"])))
                        })
                    generation_counts.setdefault(new_gen, {"LT-HSC": 0, "ST-HSC": 0, "MPP": 0})
                    generation_counts[new_gen]["ST-HSC"] += 2

                # asymmetric → 1 ST + 1 MPP
                elif fate < params["P1B"] + params["P2B"]:
                    st_next.append({
                        "gen": new_gen,
                        "timer": max(2, int(st_rng.normal(
                            params["ST_HSC_TIME_MEAN"],
                            params["ST_HSC_TIME_STD"])))
                    })
                    new_mpp[new_gen] = new_mpp.get(new_gen, 0) + 1
                    generation_counts.setdefault(new_gen, {"LT-HSC": 0, "ST-HSC": 0, "MPP": 0})
                    generation_counts[new_gen]["ST-HSC"] += 1
                    generation_counts[new_gen]["MPP"] += 1

                # symmetric diff → MPP
                elif fate < params["P1B"] + params["P2B"] + params["P3B"]:
                    new_mpp[new_gen] = new_mpp.get(new_gen, 0) + 1
                    generation_counts.setdefault(new_gen, {"LT-HSC": 0, "ST-HSC": 0, "MPP": 0})
                    generation_counts[new_gen]["MPP"] += 1

                # full diff → 2 MPP
                else:
                    new_mpp[new_gen] = new_mpp.get(new_gen, 0) + 2
                    generation_counts.setdefault(new_gen, {"LT-HSC": 0, "ST-HSC": 0, "MPP": 0})
                    generation_counts[new_gen]["MPP"] += 2
            else:
                # Still alive, already counted
                st_next.append(cell)

        lt_cells = new_lt
        st_cells = st_next

    return generation_counts


# =========================================================
# MULTI ROOT DRIVER 
# =========================================================
def run_simulation(n_roots, lt_start, st_start, steps):

    print("\nStarted:", time.ctime(), "\n")
    results = {}

    for r in tqdm(range(n_roots), desc="Root cells"):
        results[r] = simulate_root(
            root_id=r,
            lt_start=lt_start,
            st_start=st_start,
            steps=steps,
            params=params,
            seed=42
        )

    print("\nFinished:", time.ctime(), "\n")
    return results
# Ensures that the same 100 unique cell IDs will be ran everytime for reproducibility. The seed number ensures reproducibility 

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    ROOTS = 100   #Number of roots being used
    LT_START = 1000
    ST_START = 1000

    # Ensure enough steps for all possible divisions
    STEPS = params["LT_HSC_DIVISION_THRESHOLD"] * int(params["LT_HSC_TIME_MEAN"] + 10)

    print("Running simulation...\n")
    data = run_simulation(ROOTS, LT_START, ST_START, STEPS)

    print("Saving results...\n")
    with open("hematopoiesis_generations_fixed.pkl", "wb") as f:
        pickle.dump(data, f)

    print("Saved data for", len(data), "root cells.")
