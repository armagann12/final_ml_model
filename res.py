"""
convert_to_excel.py
Export all predictions and results to a readable Excel file.

Run:
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 convert_to_excel.py

Output:
    results.xlsx  — Excel file with 4 sheets
"""

import pickle
import numpy as np
import pandas as pd

# ── Load all data ─────────────────────────────────────────────────
with open("preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

with open("ensemble_results.pkl", "rb") as f:
    results = pickle.load(f)

with open("split_data.pkl", "rb") as f:
    split = pickle.load(f)


# ─────────────────────────────────────────────
# SHEET 1: TEST SET (ground truth vs prediction)
# ─────────────────────────────────────────────
test_meta  = results["test_data"]
test_probs = results["test_probs"]
y_test     = results["y_test"]
test_preds = results["test_preds"]

sheet1 = pd.DataFrame({
    "gname":          test_meta["gname"].values,
    "glong":          test_meta["glong"].values,
    "glat":           test_meta["glat"].values,
    "true_label":     ["N" if y == 1 else "F" for y in y_test],
    "predicted_label":["N" if p == 1 else "F" for p in test_preds],
    "prob_near":      np.round(test_probs, 4),
    "prob_far":       np.round(1 - test_probs, 4),
    "correct":        ["YES" if p == y else "NO"
                       for p, y in zip(test_preds, y_test)],
    "quality_factor": test_meta["kdar_manual_qf"].values,
})


# ─────────────────────────────────────────────
# SHEET 2: ALL LABELED N/F SOURCES
# ─────────────────────────────────────────────
nf_mask = data["kdar_manual"].isin(["N", "F"])
nf_data = data[nf_mask].copy().reset_index(drop=True)

# Figure out which set each source belongs to
test_names  = set(results["test_data"]["gname"])
# We don't have train/val names saved directly, label as train_val
sheet2 = pd.DataFrame({
    "gname":         nf_data["gname"],
    "glong":         nf_data["glong"],
    "glat":          nf_data["glat"],
    "rrl_velocity":  nf_data["rrl_velocity"],
    "tp_velocity":   nf_data["tp_velocity"],
    "true_label":    nf_data["kdar_manual"],
    "quality_factor":nf_data["kdar_manual_qf"],
    "split":         ["test" if n in test_names else "train/val"
                      for n in nf_data["gname"]],
})


# ─────────────────────────────────────────────
# SHEET 3: T SOURCES WITH MODEL PREDICTIONS
# ─────────────────────────────────────────────
t_meta  = results["t_data"]
t_probs = results["t_probs"]

sheet3 = pd.DataFrame({
    "gname":           t_meta["gname"].values,
    "glong":           t_meta["glong"].values,
    "glat":            t_meta["glat"].values,
    "true_label":      "T",
    "model_pred":      ["N" if p >= 0.5 else "F" for p in t_probs],
    "prob_near":       np.round(t_probs, 4),
    "prob_far":        np.round(1 - t_probs, 4),
    "quality_factor":  t_meta["kdar_manual_qf"].values,
    "note":            "Tangent — prediction is physically unreliable",
})


# ─────────────────────────────────────────────
# SHEET 4: UNLABELED SOURCES WITH PREDICTIONS
# ─────────────────────────────────────────────
unlab_meta  = results["unlabeled_data"]
unlab_probs = results["unlabeled_probs"]

sheet4 = pd.DataFrame({
    "gname":      unlab_meta["gname"].values,
    "glong":      unlab_meta["glong"].values,
    "glat":       unlab_meta["glat"].values,
    "predicted_label": ["N" if p >= 0.5 else "F" for p in unlab_probs],
    "prob_near":  np.round(unlab_probs, 4),
    "prob_far":   np.round(1 - unlab_probs, 4),
    "confidence": ["HIGH" if abs(p - 0.5) > 0.3 else "LOW"
                   for p in unlab_probs],
})


# ─────────────────────────────────────────────
# WRITE TO EXCEL
# ─────────────────────────────────────────────
output_path = "results.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    sheet1.to_excel(writer, sheet_name="Test Set (Ground Truth)", index=False)
    sheet2.to_excel(writer, sheet_name="All NF Labeled Sources",  index=False)
    sheet3.to_excel(writer, sheet_name="Tangent Sources",         index=False)
    sheet4.to_excel(writer, sheet_name="Unlabeled Predictions",   index=False)

print(f"Saved: {output_path}")
print(f"  Sheet 1 — Test Set:             {len(sheet1)} rows")
print(f"  Sheet 2 — All N/F Labeled:      {len(sheet2)} rows")
print(f"  Sheet 3 — Tangent Sources:      {len(sheet3)} rows")
print(f"  Sheet 4 — Unlabeled Predictions:{len(sheet4)} rows")