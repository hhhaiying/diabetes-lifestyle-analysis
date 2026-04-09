"""
This script merges the main Mexico Diabetes dataset with a separate physical activity
module (ENSANUT 2024) using a shared identifier (folio_int). It cleans and standardizes
the ID column in both datasets before performing a left join, then tracks how many
individuals were successfully linked.
"""

import numpy as np
import pandas as pd

main_path = "Diabetes_Mexico.csv"
sub_path  = "actividad_fisica_ensanut2024.csv"

cols_interest = ["fa0400", "fa0401", "fa0403", "fa0405", "fa0407h", "fa0407m"]

def find_id_col(cols):
    for c in ["folio_int", "FOLIO_INT"]:
        if c in cols:
            return c
    lower_map = {c.lower(): c for c in cols}
    if "folio_int" in lower_map:
        return lower_map["folio_int"]
    raise ValueError("Could not find folio_int/FOLIO_INT in columns.")

def clean_id(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return out

main_cols = pd.read_csv(main_path, nrows=0).columns
sub_cols  = pd.read_csv(sub_path,  nrows=0).columns
id_main = find_id_col(main_cols)
id_sub  = find_id_col(sub_cols)

main = pd.read_csv(main_path, low_memory=False)
sub  = pd.read_csv(sub_path, usecols=[id_sub] + cols_interest, low_memory=False)

main["_id"] = clean_id(main[id_main])
sub["_id"]  = clean_id(sub[id_sub])

main = main[main["_id"].notna()].copy()
sub  = sub[sub["_id"].notna()].copy()

dup = sub["_id"].duplicated(keep=False)
if dup.any():
    print("WARNING: subdataset has duplicated _id rows:", int(dup.sum()))
    sub = sub.sort_values("_id").drop_duplicates("_id", keep="first")

joined = main.merge(
    sub[["_id"] + cols_interest],
    on="_id",
    how="left",
    validate="m:1"
)

joined_out = joined.drop(columns=["_id"])

print("joined shape:", joined_out.shape)
print("matched rows:", int(joined[cols_interest].notna().any(axis=1).sum()))

joined_out.to_csv("Diabetes_Mexico_with_lifestyle.csv", index=False)