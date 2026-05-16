"""
step2_split_and_scale.py

Split the preprocessed data into train, validation, and test sets,
then scale the input channels.

Strategy:
- Only use N and F labeled sources for training/val/test
- T and unlabeled sources are set aside (predicted later)
- Test set uses ONLY quality A labels for reliable evaluation
- Train/val use A, B, and C quality labels
- Stratified split preserves N/F ratio across all sets
- Scale channels 0-2 (spectrum, rms, snr) using RobustScaler
- Channel 3 (id) is NOT scaled — values are already meaningful (-1 to 1)

Run:
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 step2_split_and_scale.py

Output:
    split_data.pkl  — dictionary with all splits ready for training
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_PATH  = "preprocessed_data.pkl"
OUTPUT_PATH = "split_data.pkl"

TEST_FRAC = 0.2   # fraction of A-quality N/F data held out for testing
VAL_FRAC  = 0.2   # fraction of remaining train data used for validation
SEED      = 42


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_preprocessed(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} rows from {path}")
    return data


# ─────────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────────
def split_data(data):
    """
    Separate the data into four groups:

    1. nf_data     — N and F labeled sources (used for train/val/test)
    2. t_data      — T labeled sources (set aside, predicted later)
    3. unlabeled   — no label (set aside, predicted later)

    Then within nf_data:
        test  — quality A only (ground truth evaluation)
        train — everything else, minus validation
        val   — 20% of train split
    """
    print(f"\n{'='*50}")
    print("SPLITTING DATA")
    print(f"{'='*50}")

    # ── Separate by label ─────────────────────────────────────────
    nf_mask         = data["kdar_manual"].isin(["N", "F"])
    t_mask          = data["kdar_manual"] == "T"
    unlabeled_mask  = data["kdar_manual"].isna()

    nf_data   = data[nf_mask].copy().reset_index(drop=True)
    t_data    = data[t_mask].copy().reset_index(drop=True)
    unlabeled = data[unlabeled_mask].copy().reset_index(drop=True)

    print(f"\nN/F labeled:  {len(nf_data)}")
    print(f"T labeled:    {len(t_data)}")
    print(f"Unlabeled:    {len(unlabeled)}")

    # ── Test set: quality A only ───────────────────────────────────
    # We want the test set to be the most trustworthy labels only
    a_mask    = nf_data["kdar_manual_qf"] == "A"
    a_data    = nf_data[a_mask].reset_index(drop=True)
    non_a_data = nf_data[~a_mask].reset_index(drop=True)

    print(f"\nQuality A (test pool): {len(a_data)}")
    print(f"Quality B+C (train/val pool): {len(non_a_data)}")

    # Stratified split of A-quality into test vs remaining
    a_train_val, test_data = train_test_split(
        a_data,
        test_size=TEST_FRAC,
        stratify=a_data["kdar_manual"],
        random_state=SEED,
    )

    # ── Train/val: A remainder + all B + all C ────────────────────
    train_val_data = pd.concat([a_train_val, non_a_data], ignore_index=True)

    train_data, val_data = train_test_split(
        train_val_data,
        test_size=VAL_FRAC,
        stratify=train_val_data["kdar_manual"],
        random_state=SEED,
    )

    # ── Report ────────────────────────────────────────────────────
    for name, df in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        n_total = len(df)
        n_n = (df["kdar_manual"] == "N").sum()
        n_f = (df["kdar_manual"] == "F").sum()
        qf_counts = df["kdar_manual_qf"].value_counts().to_dict()
        print(f"\n{name} set: {n_total} total | N={n_n} F={n_f} | QF={qf_counts}")

    return train_data, val_data, test_data, t_data, unlabeled


# ─────────────────────────────────────────────
# BUILD TENSORS
# ─────────────────────────────────────────────
def build_tensors(df):
    """
    Stack the 4 preprocessed channels into a (N, 301, 4) array.
    Returns X (features) and y (labels as 0=F, 1=N).
    """
    X = np.stack([
        np.stack([
            row["prep_spectrum"],
            row["prep_rms"],
            row["prep_snr"],
            row["prep_id"],
        ], axis=-1)
        for _, row in df.iterrows()
    ])

    # Encode labels: N=1, F=0
    y = (df["kdar_manual"] == "N").astype(int).values

    return X, y


def build_tensors_unlabeled(df):
    """Build tensors for sources without labels (T or unlabeled)."""
    X = np.stack([
        np.stack([
            row["prep_spectrum"],
            row["prep_rms"],
            row["prep_snr"],
            row["prep_id"],
        ], axis=-1)
        for _, row in df.iterrows()
    ])
    return X


# ─────────────────────────────────────────────
# SCALE
# ─────────────────────────────────────────────
def scale_data(X_train, X_val, X_test, X_t, X_unlabeled):
    """
    Scale channels 0, 1, 2 (spectrum, rms, snr) using RobustScaler.
    Channel 3 (id) is left untouched — already in [-1, 1].

    RobustScaler uses median and IQR instead of mean and std,
    so it's not thrown off by the bright/noisy outlier spectra.

    Scaler is fit ONLY on training data, then applied to all sets.
    """
    print(f"\n{'='*50}")
    print("SCALING")
    print(f"{'='*50}")

    n_bins = X_train.shape[1]
    channels_to_scale = [0, 1, 2]  # spectrum, rms, snr — NOT id

    # Reshape to 2D for sklearn: (N * bins, channels)
    def reshape_for_scaler(X):
        return X[:, :, :3].reshape(-1, 3)

    scaler = RobustScaler()
    scaler.fit(reshape_for_scaler(X_train))

    def apply_scale(X):
        X = X.copy()
        orig_shape = X[:, :, :3].shape
        scaled = scaler.transform(reshape_for_scaler(X))
        X[:, :, :3] = scaled.reshape(orig_shape)
        return X

    X_train_s = apply_scale(X_train)
    X_val_s   = apply_scale(X_val)
    X_test_s  = apply_scale(X_test)
    X_t_s     = apply_scale(X_t)
    X_unlab_s = apply_scale(X_unlabeled)

    print(f"Scaler fit on training data only")
    print(f"Channels scaled: spectrum, rms, snr")
    print(f"Channel NOT scaled: id (already in [-1, 1])")

    # Verify id channel unchanged
    assert np.allclose(X_train_s[:, :, 3], X_train[:, :, 3]), "ID channel was modified!"
    print(f"ID channel integrity check passed ✓")

    return scaler, X_train_s, X_val_s, X_test_s, X_t_s, X_unlab_s


# ─────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────
def compute_class_weights(y_train):
    """
    Compute class weights to handle N/F imbalance.
    Weight = total_samples / (n_classes * class_count)
    So the minority class (N) gets a higher weight.
    """
    n_total = len(y_train)
    n_f = (y_train == 0).sum()
    n_n = (y_train == 1).sum()

    weight_f = n_total / (2 * n_f)
    weight_n = n_total / (2 * n_n)

    print(f"\n{'='*50}")
    print("CLASS WEIGHTS")
    print(f"{'='*50}")
    print(f"N count: {n_n}  weight: {weight_n:.3f}")
    print(f"F count: {n_f}  weight: {weight_f:.3f}")
    print(f"(Higher weight = more influence during training)")

    return {0: weight_f, 1: weight_n}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load preprocessed data
    data = load_preprocessed(INPUT_PATH)

    # 2. Split into groups
    train_data, val_data, test_data, t_data, unlabeled = split_data(data)

    # 3. Build input tensors
    print(f"\n{'='*50}")
    print("BUILDING TENSORS")
    print(f"{'='*50}")
    X_train, y_train = build_tensors(train_data)
    X_val,   y_val   = build_tensors(val_data)
    X_test,  y_test  = build_tensors(test_data)
    X_t               = build_tensors_unlabeled(t_data)
    X_unlabeled       = build_tensors_unlabeled(unlabeled)

    print(f"X_train shape: {X_train.shape}  y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}  y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}  y_test shape:  {y_test.shape}")
    print(f"X_t shape:     {X_t.shape}  (T sources, predicted later)")
    print(f"X_unlabeled:   {X_unlabeled.shape}  (unlabeled, predicted later)")

    # 4. Scale
    scaler, X_train, X_val, X_test, X_t, X_unlabeled = scale_data(
        X_train, X_val, X_test, X_t, X_unlabeled
    )

    # 5. Class weights
    class_weights = compute_class_weights(y_train)

    # 6. Save everything
    output = {
        # Tensors
        "X_train": X_train,
        "y_train": y_train,
        "X_val":   X_val,
        "y_val":   y_val,
        "X_test":  X_test,
        "y_test":  y_test,
        "X_t":     X_t,
        "X_unlabeled": X_unlabeled,

        # Metadata for predictions later
        "t_data":       t_data[["gname", "glong", "glat", "kdar_manual", "kdar_manual_qf"]].reset_index(drop=True),
        "unlabeled_data": unlabeled[["gname", "glong", "glat"]].reset_index(drop=True),
        "test_data":    test_data[["gname", "glong", "glat", "kdar_manual", "kdar_manual_qf"]].reset_index(drop=True),

        # Scaler and weights
        "scaler":        scaler,
        "class_weights": class_weights,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(output, f)

    print(f"\n{'='*50}")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Step 2 complete. Run step3_train.py next.")
    print(f"{'='*50}")