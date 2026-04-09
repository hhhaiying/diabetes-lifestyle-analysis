"""
This script prepares the main analytic sample from the merged Mexico Diabetes dataset.
It filters to complete lifestyle cases, cleans and validates key variables (age, sex,
weight, height, BMI), fills missing BMI from weight and height where possible, and
drops rows missing core covariates. A binary high-risk outcome is derived from the
diabetes risk category. The final analytic dataset is renamed to English column names
and saved as a CSV for use in modelling.
"""

import pandas as pd

input_path = "Diabetes_Mexico_with_lifestyle_recode.csv"
output_path = "Diabetes_Mexico_analysis.csv"

df = pd.read_csv(input_path, low_memory=False)

required_cols = [
    "folio_i",
    "folio_int",
    "riesgo_diabetes_cat",
    "edad",
    "sexo",
    "Ciudad",
    "sleep_cat",
    "vig_days",
    "mod_days",
    "walk_days",
    "sedentary_min",
    "lifestyle_complete_after_recode",
    "Peso",
    "Estatura",
    "imc"
]

for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

# Main analytic sample
analysis_df = df[df["lifestyle_complete_after_recode"] == True].copy()

# Clean sex
analysis_df["sexo_raw"] = analysis_df["sexo"]
analysis_df["sexo"] = analysis_df["sexo"].astype(str).str.strip()

sex_map = {
    "Hombre": 1,
    "Mujer": 2
}

analysis_df["sexo"] = analysis_df["sexo"].replace(sex_map)
analysis_df["sexo"] = pd.to_numeric(analysis_df["sexo"], errors="coerce")

# Clean age
analysis_df["edad_raw"] = analysis_df["edad"]
analysis_df["edad"] = pd.to_numeric(analysis_df["edad"], errors="coerce")
analysis_df.loc[~analysis_df["edad"].between(0, 120), "edad"] = pd.NA

# Convert numeric columns
other_num_cols = [
    "riesgo_diabetes_cat",
    "sleep_cat",
    "vig_days",
    "mod_days",
    "walk_days",
    "sedentary_min",
    "Peso",
    "Estatura",
    "imc"
]

for c in other_num_cols:
    analysis_df[c] = pd.to_numeric(analysis_df[c], errors="coerce")

# Basic plausibility checks
analysis_df.loc[~analysis_df["Peso"].between(20, 300), "Peso"] = pd.NA
analysis_df.loc[~analysis_df["Estatura"].between(100, 250), "Estatura"] = pd.NA
analysis_df.loc[~analysis_df["imc"].between(10, 80), "imc"] = pd.NA

# Fill missing BMI from weight and height when possible
height_m = analysis_df["Estatura"] / 100
bmi_from_hw = analysis_df["Peso"] / (height_m ** 2)

can_fill_bmi = (
    analysis_df["imc"].isna() &
    analysis_df["Peso"].notna() &
    analysis_df["Estatura"].notna() &
    height_m.notna() &
    (height_m > 0)
)

analysis_df.loc[can_fill_bmi, "imc"] = bmi_from_hw[can_fill_bmi]

# Re-check BMI range after filling
analysis_df.loc[~analysis_df["imc"].between(10, 80), "imc"] = pd.NA

print("BMI missing after fill:", analysis_df["imc"].isna().sum())

# Only drop truly key variables for current analysis
# Do NOT drop rows just because weight / height / BMI are missing
analysis_df = analysis_df.dropna(
    subset=[
        "riesgo_diabetes_cat",
        "edad",
        "sexo",
        "Ciudad"
    ]
).copy()

# Binary outcome only
analysis_df["high_risk"] = (analysis_df["riesgo_diabetes_cat"] == 2).astype(int)

print("\n=== MAIN ANALYTIC SAMPLE ===")
print("n:", len(analysis_df))

print("\n=== MISSING CHECK IN ANALYTIC SAMPLE ===")
print(
    analysis_df[[
        "riesgo_diabetes_cat", "edad", "sexo", "Ciudad",
        "Peso", "Estatura", "imc"
    ]].isna().sum()
)

print("\n=== BINARY OUTCOME DISTRIBUTION ===")
outcome_counts = analysis_df["high_risk"].value_counts(dropna=False).sort_index()
outcome_props = analysis_df["high_risk"].value_counts(normalize=True, dropna=False).sort_index()
print(pd.DataFrame({"count": outcome_counts, "prop": outcome_props}))

# Rename columns to English for exported file
analysis_df = analysis_df.rename(columns={
    "riesgo_diabetes_cat": "diabetes_risk_cat",
    "edad": "age",
    "sexo": "sex",
    "Ciudad": "city",
    "Peso": "weight_kg",
    "Estatura": "height_cm",
    "imc": "bmi"
})

# Save clean analysis file
keep_cols = [
    "folio_i",
    "folio_int",
    "diabetes_risk_cat",
    "high_risk",
    "age",
    "sex",
    "city",
    "sleep_cat",
    "vig_days",
    "mod_days",
    "walk_days",
    "sedentary_min",
    "weight_kg",
    "height_cm",
    "bmi"
]

analysis_df[keep_cols].to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")