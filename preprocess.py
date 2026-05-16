import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH = "wise_hi_absorption_8.5arcmin_manual.pkl"
OUTPUT_PATH = "preprocessed_data.pkl"

# Fixed velocity axis every sample will be resampled onto
VELOCITY_MIN = -100.0   # km/s
VELOCITY_MAX = 200.0    # km/s
VELOCITY_STEP = 1.0     # km/s
NEW_VELOCITY = np.arange(VELOCITY_MIN, VELOCITY_MAX + VELOCITY_STEP, VELOCITY_STEP)


# ─────────────────────────────────────────────
# STEP 1A: LOAD
# ─────────────────────────────────────────────
def load_data(path):
    """Load the raw pickle file and return a DataFrame."""
    print(f"\n{'='*50}")
    print("LOADING DATA")
    print(f"{'='*50}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"Total rows loaded: {len(data)}")
    print(f"Columns: {list(data.columns)}")
    return data


# ─────────────────────────────────────────────
# STEP 1B: EXPLORE
# ─────────────────────────────────────────────
def explore_data(data):
    """Print key statistics to understand the dataset before doing anything."""
    print(f"\n{'='*50}")
    print("DATA EXPLORATION")
    print(f"{'='*50}")

    # Label distribution
    print("\n--- Label distribution (kdar_manual) ---")
    print(data["kdar_manual"].value_counts(dropna=False))

    # Quality factor distribution
    print("\n--- Quality factor distribution (kdar_manual_qf) ---")
    print(data["kdar_manual_qf"].value_counts(dropna=False))

    # Label x Quality factor cross-tab
    print("\n--- Label x Quality Factor ---")
    labeled = data[data["kdar_manual"].notna()]
    print(pd.crosstab(labeled["kdar_manual"], labeled["kdar_manual_qf"], margins=True))

    # Unlabeled count
    n_unlabeled = data["kdar_manual"].isna().sum()
    n_labeled = data["kdar_manual"].notna().sum()
    print(f"\nLabeled:   {n_labeled}")
    print(f"Unlabeled: {n_unlabeled}")
    print(f"Total:     {len(data)}")

    # Velocity axis shape check
    sample_vel = data["velocity"].iloc[0]
    sample_spec = data["spectrum"].iloc[0]
    print(f"\n--- Spectrum shape check (first row) ---")
    print(f"Velocity axis length: {len(sample_vel)}")
    print(f"Spectrum length:      {len(sample_spec)}")
    print(f"Velocity range:       {sample_vel.min():.1f} to {sample_vel.max():.1f} km/s")

    # Longitude distribution (to see 4th quadrant proportion)
    print(f"\n--- Galactic longitude stats ---")
    print(f"Min glong: {data['glong'].min():.1f} deg")
    print(f"Max glong: {data['glong'].max():.1f} deg")
    n_4th = (data["glong"] > 270.0).sum()
    print(f"4th quadrant (glong > 270): {n_4th} sources")

    # Check for missing rrl_velocity or tp_velocity
    print(f"\n--- Missing values ---")
    for col in ["rrl_velocity", "tp_velocity", "glong", "glat"]:
        n_missing = data[col].isna().sum()
        print(f"{col}: {n_missing} missing")


# ─────────────────────────────────────────────
# STEP 1C: PREPROCESSING HELPERS
# ─────────────────────────────────────────────
def resample_spectrum(spectrum, old_velocity, new_velocity):
    """
    Interpolate a spectrum onto a new velocity axis.
    Uses linear interpolation; returns 0 outside the original range.
    """
    interp = interp1d(
        old_velocity,
        spectrum,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return interp(new_velocity)


def make_id_spectrum(new_velocity, rrl_velocity, tp_velocity):
    """
    Create the physics-informed ID channel.

    Encodes WHERE in velocity space each bin sits relative to the
    physically meaningful boundaries V_R (HII region) and V_T (tangent point):

        velocity <= 0              → -1.0
        0 < velocity <= V_R        → scales linearly from -1 to 0
        V_R < velocity < V_T       → scales linearly from 0 to 1
        velocity >= V_T            → 1.0

    Interpretation:
    - A source at the FAR distance shows absorption up through the V_R-to-V_T zone (ID > 0)
    - A source at the NEAR distance shows absorption only up to V_R (ID < 0)
    - This bakes the key physics directly into the input
    """
    prep_id = np.zeros_like(new_velocity)

    prep_id[new_velocity <= 0.0] = -1.0

    mask = (new_velocity > 0.0) & (new_velocity <= rrl_velocity)
    if rrl_velocity > 0:
        prep_id[mask] = new_velocity[mask] / rrl_velocity - 1.0

    mask = (new_velocity > rrl_velocity) & (new_velocity < tp_velocity)
    dv = tp_velocity - rrl_velocity
    if dv > 0:
        prep_id[mask] = (new_velocity[mask] - rrl_velocity) / dv

    prep_id[new_velocity >= tp_velocity] = 1.0

    return prep_id


def make_snr_spectrum(spectrum, rms):
    """
    Compute signal-to-noise ratio per velocity bin.

    SNR = spectrum / rms

    This tells the network which parts of the spectrum are reliable
    (high SNR) versus noise-dominated (low SNR). Clips to avoid
    extreme values and replaces NaN/inf with 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(rms > 0, spectrum / rms, 0.0)
    snr = np.clip(snr, -10.0, 10.0)
    snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
    return snr


# ─────────────────────────────────────────────
# STEP 1D: PREPROCESS EACH ROW
# ─────────────────────────────────────────────
def preprocess_row(row, new_velocity):
    """
    Preprocess a single row of the DataFrame.

    Returns a dict with:
        prep_spectrum : resampled brightness spectrum  [channel 0]
        prep_rms      : resampled noise spectrum       [channel 1]
        prep_snr      : signal-to-noise spectrum       [channel 2]  ← NEW
        prep_id       : physics-informed ID spectrum   [channel 3]
    """
    old_velocity = row["velocity"].copy()
    spectrum = row["spectrum"].copy()
    rms = row["rms"].copy()
    rrl_velocity = float(row["rrl_velocity"])
    tp_velocity = float(row["tp_velocity"])

    # ── 4th quadrant flip ──────────────────────────────────────────
    # In the 4th Galactic quadrant (glong > 270°), circular rotation
    # causes velocities to be negative. We flip the axis so all
    # spectra look geometrically the same regardless of quadrant.
    if row["glong"] > 270.0:
        old_velocity = -1.0 * old_velocity[::-1]
        spectrum = spectrum[::-1]
        rms = rms[::-1]
        rrl_velocity = -rrl_velocity
        tp_velocity = -tp_velocity

    # ── Resample onto fixed velocity axis ─────────────────────────
    prep_spectrum = resample_spectrum(spectrum, old_velocity, new_velocity)
    prep_rms = resample_spectrum(rms, old_velocity, new_velocity)

    # ── Replace NaN/inf from resampling ───────────────────────────
    prep_spectrum = np.nan_to_num(prep_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
    prep_rms = np.nan_to_num(prep_rms, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Compute SNR channel ───────────────────────────────────────
    prep_snr = make_snr_spectrum(prep_spectrum, prep_rms)

    # ── Compute ID channel ────────────────────────────────────────
    prep_id = make_id_spectrum(new_velocity, rrl_velocity, tp_velocity)

    return {
        "prep_spectrum": prep_spectrum,
        "prep_rms": prep_rms,
        "prep_snr": prep_snr,
        "prep_id": prep_id,
    }


def preprocess_data(data, new_velocity):
    """
    Apply preprocessing to every row and add result columns to the DataFrame.

    Output channels per source (each is a 1D array of length = len(new_velocity)):
        Channel 0: prep_spectrum  — raw brightness
        Channel 1: prep_rms      — noise level
        Channel 2: prep_snr      — signal-to-noise ratio  (NEW vs professor)
        Channel 3: prep_id       — physics ID encoding
    """
    print(f"\n{'='*50}")
    print("PREPROCESSING")
    print(f"{'='*50}")
    print(f"Fixed velocity axis: {new_velocity.min():.0f} to {new_velocity.max():.0f} km/s "
          f"in steps of {VELOCITY_STEP:.0f} km/s → {len(new_velocity)} bins")

    results = []
    for i, (idx, row) in enumerate(data.iterrows()):
        if i % 100 == 0:
            print(f"  Processing row {i}/{len(data)}...")
        result = preprocess_row(row, new_velocity)
        results.append(result)

    results_df = pd.DataFrame(results, index=data.index)
    data = pd.concat([data, results_df], axis=1)

    print(f"\nPreprocessing complete.")
    print(f"New columns added: {list(results_df.columns)}")
    print(f"Each column is an array of length {len(new_velocity)}")
    print(f"\nFinal input shape per source: ({len(new_velocity)}, 4)")
    print(f"  Axis 0: {len(new_velocity)} velocity bins")
    print(f"  Axis 1: 4 channels [spectrum, rms, snr, id]")

    return data


# ─────────────────────────────────────────────
# STEP 1E: POST-PROCESSING CHECKS
# ─────────────────────────────────────────────
def post_checks(data):
    """Sanity checks after preprocessing."""
    print(f"\n{'='*50}")
    print("POST-PREPROCESSING CHECKS")
    print(f"{'='*50}")

    # Check for NaN in any preprocessed channel
    for col in ["prep_spectrum", "prep_rms", "prep_snr", "prep_id"]:
        n_nan = sum(np.any(np.isnan(arr)) for arr in data[col])
        print(f"Rows with NaN in {col}: {n_nan}")

    # Check value ranges
    for col in ["prep_spectrum", "prep_rms", "prep_snr", "prep_id"]:
        all_vals = np.concatenate(data[col].values)
        print(f"{col:18s}  min={all_vals.min():.3f}  max={all_vals.max():.3f}  "
              f"mean={all_vals.mean():.3f}")

    # Confirm shape
    sample = np.stack([
        data["prep_spectrum"].iloc[0],
        data["prep_rms"].iloc[0],
        data["prep_snr"].iloc[0],
        data["prep_id"].iloc[0],
    ], axis=-1)
    print(f"\nSample input tensor shape: {sample.shape}  ✓")


# ─────────────────────────────────────────────
# STEP 1F: BUILD FINAL TENSOR FUNCTION
# ─────────────────────────────────────────────
def build_input_tensor(row):
    """
    Convert a preprocessed DataFrame row into a (num_bins, 4) tensor.
    This is what you will feed into the neural network.

    Channel order:
        [:, 0] = spectrum
        [:, 1] = rms
        [:, 2] = snr
        [:, 3] = id
    """
    return np.stack([
        row["prep_spectrum"],
        row["prep_rms"],
        row["prep_snr"],
        row["prep_id"],
    ], axis=-1)


def build_all_tensors(data):
    """
    Build the full input tensor X of shape (N, num_bins, 4).
    Also returns the label array and quality factor array.
    """
    X = np.stack([build_input_tensor(row) for _, row in data.iterrows()])
    labels = data["kdar_manual"].values         # N, F, T, or NaN
    qf = data["kdar_manual_qf"].values          # A, B, C, or NaN
    return X, labels, qf


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load
    data = load_data(DATA_PATH)

    # 2. Explore — understand the data before touching it
    explore_data(data)

    # 3. Preprocess — add 4 prepared channels to each row
    data = preprocess_data(data, NEW_VELOCITY)

    # 4. Sanity checks
    post_checks(data)

    # 5. Save preprocessed DataFrame for use in next steps
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved preprocessed data to: {OUTPUT_PATH}")

    # 6. Quick demo — build the full tensor to confirm shape
    print(f"\n{'='*50}")
    print("BUILDING FULL INPUT TENSOR (demo)")
    print(f"{'='*50}")
    X, labels, qf = build_all_tensors(data)
    print(f"X shape:      {X.shape}   (sources, velocity_bins, channels)")
    print(f"Labels shape: {labels.shape}")
    print(f"QF shape:     {qf.shape}")

    labeled_mask = ~pd.isna(labels)
    print(f"\nLabeled subset:   {labeled_mask.sum()} sources")
    print(f"Unlabeled subset: {(~labeled_mask).sum()} sources")
    print(f"\nStep 1 complete. Run step2_split_and_scale.py next.")