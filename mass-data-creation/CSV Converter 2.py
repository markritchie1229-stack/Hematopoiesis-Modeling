import pickle
import pandas as pd

# --- Load the per-generation simulation data ---
with open("hematopoiesis_generations_fixed.pkl", "rb") as f:
    generation_data = pickle.load(f)

# --- Prepare table for per-generation population sizes ---
rows = []

for root_id, gen_dict in generation_data.items():
    for gen in sorted(gen_dict.keys()):
        counts = gen_dict[gen]
        rows.append({
            "Root_ID": root_id,
            "Generation": gen,
            "LT-HSC_Count": counts["LT-HSC"],
            "ST-HSC_Count": counts["ST-HSC"],
            "MPP_Count": counts["MPP"]
        })

# --- Create DataFrame ---
df = pd.DataFrame(rows)

# --- Save to CSV ---
df.to_csv("hematopoiesis_generation_sizes.csv", index=False)

print(f"CSV saved with {len(df)} rows for {len(generation_data)} root cells.")