import pandas as pd
import matplotlib.pyplot as plt
import os

print(os.listdir("/Users/markritchie/Documents"))
# ======================================================
# SETTINGS — CHANGE IF YOUR COLUMN NAMES DIFFER
# ======================================================
CSV_FILE = "/Users/markritchie/Documents/Newest Hemotioiesis Lineage Tree Folder/hematopoiesis_generation_sizes.csv"

COL_ID  = "Root_ID"
COL_GEN = "Generation"
COL_LT  = "LT-HSC_Count"
COL_ST  = "ST-HSC_Count"
COL_MPP = "MPP_Count"

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(CSV_FILE)

# sort for clean line plotting
df = df.sort_values([COL_ID, COL_GEN])

# unique cell IDs
cell_ids = df[COL_ID].unique()

# ======================================================
# FUNCTION TO PLOT ONE CELL TYPE
# ======================================================
def plot_type(column, title):
    plt.figure()

    for cid in cell_ids:
        sub = df[df[COL_ID] == cid]
        plt.plot(sub[COL_GEN], sub[column])

    plt.xlabel("Generation")
    plt.ylabel("Population")
    plt.title(title)
    plt.grid(True)

# ======================================================
# CREATE COMBINED COLUMN
# ======================================================
df["TOTAL"] = df[COL_LT] + df[COL_ST] + df[COL_MPP]

# ======================================================
# PLOTS
# ======================================================
plot_type(COL_ST,  "ST Cells vs Generation")
plot_type(COL_LT,  "LT Cells vs Generation")
plot_type(COL_MPP, "MPP Cells vs Generation")
plot_type("TOTAL", "Total Cells vs Generation")

# ======================================================
# SHOW ALL FIGURES
# ======================================================
plt.show()

