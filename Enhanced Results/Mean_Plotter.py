# Calculates and plots the mean value of number of LT Cells, ST Cells, and total number of cells over all 100 unique ids for each generation. 
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# --- Load the per-generation simulation data ---
with open("hematopoiesis_generations_fixed.pkl", "rb") as f:
    generation_data = pickle.load(f)

# --- Prepare table for per-generation population sizes ---
rows = []

for root_id, gen_dict in generation_data.items():
    for gen in sorted(gen_dict.keys()):
        counts = gen_dict[gen]
        total_count = counts["LT-HSC"] + counts["ST-HSC"] + counts["MPP"]
        rows.append({
            "Root_ID": root_id,
            "Generation": gen,
            "LT-HSC_Count": counts["LT-HSC"],
            "ST-HSC_Count": counts["ST-HSC"],
            "Total_Count": total_count
        })

# --- Create DataFrame ---
df = pd.DataFrame(rows)

# --- Aggregate by Generation across all root IDs ---
agg_df = df.groupby("Generation").mean().reset_index()

# --- Plot LT-HSC counts ---
plt.figure(figsize=(8,5))
plt.plot(agg_df["Generation"], agg_df["LT-HSC_Count"], marker='o')
plt.title("Mean LT-HSC Count per Generation")
plt.xlabel("Generation")
plt.ylabel("Mean LT-HSC Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot ST-HSC counts ---
plt.figure(figsize=(8,5))
plt.plot(agg_df["Generation"], agg_df["ST-HSC_Count"], marker='o', color='orange')
plt.title("Mean ST-HSC Count per Generation")
plt.xlabel("Generation")
plt.ylabel("Mean ST-HSC Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Total counts ---
plt.figure(figsize=(8,5))
plt.plot(agg_df["Generation"], agg_df["Total_Count"], marker='o', color='green')
plt.title("Mean Total Cell Count per Generation")
plt.xlabel("Generation")
plt.ylabel("Mean Total Cell Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Optional: Save aggregated data to CSV ---
agg_df.to_csv("hematopoiesis_generation_means.csv", index=False)
print(f"Aggregated CSV saved with {len(agg_df)} generations.")
