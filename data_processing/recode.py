"""
This script recodes lifestyle variables from the Mexico Diabetes dataset.
It processes six variables: sleep duration (fa0400), vigorous/moderate/walking activity days
(fa0401, fa0403, fa0405), and sedentary time (fa0407h, fa0407m).
Missing/non-response codes (88, 99) are replaced with NaN, and inability-to-move flags (55)
are captured separately. Sleep categories are mapped to readable labels, physical activity
days are cleaned into model-ready variables, and sedentary hours/minutes are combined into
a single total in minutes. A completeness flag is added to identify rows with full data
across all lifestyle variables. The recoded dataset is saved as a new CSV file.
"""

import numpy as np
import pandas as pd

input_path = "Diabetes_Mexico_with_lifestyle.csv"
output_path = "Diabetes_Mexico_with_lifestyle_recode.csv"

cols_interest = ["fa0400", "fa0401", "fa0403", "fa0405", "fa0407h", "fa0407m"]
pa_vars = ["fa0401", "fa0403", "fa0405"]

df = pd.read_csv(input_path, low_memory=False)

# Age filter: keep only valid ages (0-120)
df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
df = df[df["edad"].between(0, 120)].copy()

for c in cols_interest:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

# Keep raw copies
for c in cols_interest:
    df[c + "_raw"] = df[c]

# Convert to numeric
for c in cols_interest:
    df[c] = pd.to_numeric(df[c], errors="coerce")

missing_codes = [88, 99]
unable_code = 55

# Sleep
df["sleep_cat"] = df["fa0400"].replace(missing_codes, np.nan)

sleep_map = {
    1: "<=5h",
    2: "6h",
    3: "7h",
    4: "8h",
    5: ">=9h"
}
df["sleep_label"] = df["sleep_cat"].map(sleep_map)

# Physical activity
for c in pa_vars:
    df[c + "_unable_flag"] = (df[c] == unable_code).astype(int)
    df[c + "_clean"] = df[c].replace(missing_codes + [unable_code], np.nan)

df["vig_days"] = df["fa0401_clean"]
df["mod_days"] = df["fa0403_clean"]
df["walk_days"] = df["fa0405_clean"]

# Sedentary time
df["fa0407h_clean"] = df["fa0407h"].replace(missing_codes, np.nan)
df["fa0407m_clean"] = df["fa0407m"].replace(missing_codes, np.nan)

df.loc[
    df["fa0407h_clean"].notna() & df["fa0407m_clean"].isna(),
    "fa0407m_clean"
] = 0

df.loc[
    df["fa0407m_clean"].notna() & ~df["fa0407m_clean"].between(0, 59),
    "fa0407m_clean"
] = np.nan

df["sedentary_min"] = df["fa0407h_clean"] * 60 + df["fa0407m_clean"]

# Complete-case flag after recoding
model_vars = ["sleep_cat", "vig_days", "mod_days", "walk_days", "sedentary_min"]
df["lifestyle_complete_after_recode"] = df[model_vars].notna().all(axis=1)

# Summary
print("\n=== RECODE SUMMARY ===")
print(pd.concat({c: df[c].value_counts(dropna=False).sort_index() 
                 for c in ["sleep_cat", "vig_days", "mod_days", "walk_days"]}, axis=1).to_string())
print("\nsedentary_min:", df["sedentary_min"].describe().round(1).to_dict())

print("\n=== UNABLE FLAGS ===")
print({c + "_unable_flag": int(df[c + "_unable_flag"].sum()) for c in pa_vars})

print("\n=== COMPLETE AFTER RECODE ===")
print(f"n={int(df['lifestyle_complete_after_recode'].sum())}, rate={df['lifestyle_complete_after_recode'].mean():.3f}")

df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")