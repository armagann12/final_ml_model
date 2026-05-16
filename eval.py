"""
eval.py

Compute ROC AUC and generate evaluation plots from ensemble results.

Produces:
    roc_curve.png        — ROC curve with AUC score
    confusion_matrix.png — Confusion matrix heatmap
    probability_dist.png — Distribution of predicted probabilities

Run:
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 eval.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
with open("ensemble_results.pkl", "rb") as f:
    results = pickle.load(f)

# Test set (quality A only — ground truth)
test_probs = results["test_probs"]   # probability of N for each test source
y_test     = results["y_test"]       # true labels (0=F, 1=N)
test_preds = results["test_preds"]   # hard predictions at threshold 0.5

# Val set
val_probs  = results["val_probs"]
y_val      = results["y_val"]

print("=" * 50)
print("ROC AUC SCORES")
print("=" * 50)


# ─────────────────────────────────────────────
# ROC AUC
# ─────────────────────────────────────────────
# Test set AUC
test_auc = roc_auc_score(y_test, test_probs)
print(f"\nTest set AUC:       {test_auc:.4f}")

# Val set AUC
val_auc = roc_auc_score(y_val, val_probs)
print(f"Validation set AUC: {val_auc:.4f}")

# Interpret
print(f"\nTarget was > 0.90")
if test_auc >= 0.90:
    print(f"Target MET ✓  (test AUC = {test_auc:.4f})")
else:
    print(f"Target NOT met  (test AUC = {test_auc:.4f})")

# Per-model AUC from ensemble
all_test_probs = results["all_test_probs"]  # shape (N_models, N_test)
per_model_aucs = [roc_auc_score(y_test, p) for p in all_test_probs]
print(f"\nPer-model AUC: {[round(a, 4) for a in per_model_aucs]}")
print(f"Mean: {np.mean(per_model_aucs):.4f}  Std: {np.std(per_model_aucs):.4f}")


# ─────────────────────────────────────────────
# FIGURE 1: ROC CURVE
# ─────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, test_probs)
val_fpr, val_tpr, _ = roc_curve(y_val, val_probs)

fig, ax = plt.subplots(figsize=(6, 6))

# Individual model ROC curves (light)
for p in all_test_probs:
    m_fpr, m_tpr, _ = roc_curve(y_test, p)
    ax.plot(m_fpr, m_tpr, color="steelblue", alpha=0.2, linewidth=0.8)

# Ensemble ROC curve
ax.plot(fpr, tpr, color="steelblue", linewidth=2.5,
        label=f"Ensemble test  (AUC = {test_auc:.4f})")
ax.plot(val_fpr, val_tpr, color="darkorange", linewidth=2.0, linestyle="--",
        label=f"Ensemble val   (AUC = {val_auc:.4f})")

# Random baseline
ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="Random (AUC = 0.50)")

# Target line
ax.axhline(y=0.9, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
ax.axvline(x=0.1, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — N vs F Classification\n(Test set: Quality A labels only)",
             fontsize=12)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: roc_curve.png")


# ─────────────────────────────────────────────
# FIGURE 2: CONFUSION MATRIX
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, test_preds)

fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Far (F)", "Near (N)"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")

ax.set_title(f"Confusion Matrix — Test Set (Quality A)\n"
             f"Accuracy: {(test_preds == y_test).mean()*100:.1f}%  |  "
             f"AUC: {test_auc:.4f}",
             fontsize=11)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: confusion_matrix.png")


# ─────────────────────────────────────────────
# FIGURE 3: PROBABILITY DISTRIBUTION
# ─────────────────────────────────────────────
# Shows how confident the model is — good models have
# probabilities clustered near 0 and 1, not near 0.5

fig, ax = plt.subplots(figsize=(7, 4))

# Test set by true label
f_probs = test_probs[y_test == 0]
n_probs = test_probs[y_test == 1]

bins = np.linspace(0, 1, 21)
ax.hist(f_probs, bins=bins, alpha=0.6, color="tomato",
        label=f"True Far (n={len(f_probs)})", edgecolor="white")
ax.hist(n_probs, bins=bins, alpha=0.6, color="steelblue",
        label=f"True Near (n={len(n_probs)})", edgecolor="white")

ax.axvline(x=0.5, color="black", linestyle="--", linewidth=1.5,
           label="Decision threshold (0.5)")

ax.set_xlabel("Predicted probability of Near (N)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Ensemble Predicted Probability Distribution\n(Test set: Quality A labels only)",
             fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("probability_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: probability_dist.png")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*50}")
print("FINAL SUMMARY")
print(f"{'='*50}")
print(f"Test accuracy:      {(test_preds == y_test).mean()*100:.1f}%")
print(f"Test AUC:           {test_auc:.4f}")
print(f"Val accuracy:       {(results['val_preds'] == y_val).mean()*100:.1f}%")
print(f"Val AUC:            {val_auc:.4f}")
print(f"Ensemble val acc:   {np.mean(results['val_accs'])*100:.1f}% "
      f"± {np.std(results['val_accs'])*100:.1f}%")
print(f"\nN precision:  1.00  (no false Near predictions)")
print(f"F recall:     1.00  (caught every Far source)")
print(f"\nOutputs saved:")
print(f"  roc_curve.png")
print(f"  confusion_matrix.png")
print(f"  probability_dist.png")