import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, models
from sklearn.metrics import confusion_matrix, classification_report

INPUT_PATH   = "split_data.pkl"
OUTPUT_PATH  = "ensemble_results.pkl"
MODELS_DIR   = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

N_MODELS      = 10
SEEDS         = list(range(N_MODELS))
NUM_BLOCKS    = 4
KERNEL_SIZE   = 7
BATCH_NORM    = True
LEARNING_RATE = 0.001
BATCH_SIZE    = 16
EPOCHS        = 200


def build_resnet(input_shape, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = keras.Input(shape=input_shape)
    x = inputs

    for i in range(NUM_BLOCKS):
        filters = 8 * (2 ** i)
        shortcut = x

        x = layers.Conv1D(filters, KERNEL_SIZE, padding="same")(x)
        if BATCH_NORM:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv1D(filters, KERNEL_SIZE, padding="same")(x)
        if BATCH_NORM:
            x = layers.BatchNormalization()(x)

        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
            if BATCH_NORM:
                shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_one(X_train, y_train, X_val, y_val, class_weights, seed, model_path):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_resnet(input_shape=X_train.shape[1:], seed=seed)

    cbs = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=0,
    )

    best_val_acc = max(history.history["val_accuracy"])
    best_epoch   = np.argmax(history.history["val_accuracy"]) + 1
    return model, best_val_acc, best_epoch


def ensemble_predict(model_paths, X):
    all_probs = []
    for path in model_paths:
        model = models.load_model(path)
        probs = model.predict(X, verbose=0).flatten()
        all_probs.append(probs)
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs, np.array(all_probs)


def evaluate(probs, y_true, label="Test", threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc   = (preds == y_true).mean()

    print(f"\n--- {label} ---")
    print(f"Accuracy: {acc*100:.1f}%  ({int(acc*len(y_true))}/{len(y_true)} correct)")

    cm = confusion_matrix(y_true, preds)
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"           Pred F   Pred N")
    print(f"True F     {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"True N     {cm[1,0]:5d}    {cm[1,1]:5d}")

    print(f"\nClassification Report:")
    print(classification_report(y_true, preds, target_names=["F", "N"]))

    return acc, preds


if __name__ == "__main__":

    print(f"{'='*50}")
    print("LOADING SPLIT DATA")
    print(f"{'='*50}")

    with open(INPUT_PATH, "rb") as f:
        d = pickle.load(f)

    X_train       = d["X_train"]
    y_train       = d["y_train"]
    X_val         = d["X_val"]
    y_val         = d["y_val"]
    X_test        = d["X_test"]
    y_test        = d["y_test"]
    X_t           = d["X_t"]
    X_unlabeled   = d["X_unlabeled"]
    class_weights = d["class_weights"]
    test_data     = d["test_data"]

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"Class weights: {class_weights}")

    print(f"\n{'='*50}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*50}")
    sample_model = build_resnet(input_shape=X_train.shape[1:], seed=0)
    sample_model.summary()

    print(f"\n{'='*50}")
    print(f"TRAINING ENSEMBLE ({N_MODELS} models)")
    print(f"{'='*50}")

    model_paths = []
    val_accs    = []

    for i, seed in enumerate(SEEDS):
        model_path = os.path.join(MODELS_DIR, f"model_{seed}.keras")
        model_paths.append(model_path)

        print(f"\nModel {i+1}/{N_MODELS}  (seed={seed})", end="  ")
        model, best_val_acc, best_epoch = train_one(
            X_train, y_train, X_val, y_val,
            class_weights, seed, model_path
        )
        val_accs.append(best_val_acc)
        print(f"best val_acc={best_val_acc*100:.1f}%  epoch={best_epoch}")

    print(f"\nEnsemble val accuracy: {np.mean(val_accs)*100:.1f}% "
          f"± {np.std(val_accs)*100:.1f}%")

    print(f"\n{'='*50}")
    print("EVALUATION ON TEST SET (Quality A only)")
    print(f"{'='*50}")

    test_probs, all_test_probs = ensemble_predict(model_paths, X_test)
    test_acc, test_preds = evaluate(test_probs, y_test, label="Ensemble on Test Set")

    print(f"\n{'='*50}")
    print("VALIDATION SET BREAKDOWN BY QUALITY FACTOR")
    print(f"{'='*50}")

    val_probs, _ = ensemble_predict(model_paths, X_val)
    val_preds    = (val_probs >= 0.5).astype(int)
    print(f"Overall val accuracy: {(val_preds == y_val).mean()*100:.1f}%")

    print(f"\n{'='*50}")
    print("PREDICTING ON T SOURCES AND UNLABELED")
    print(f"{'='*50}")

    t_probs, _         = ensemble_predict(model_paths, X_t)
    unlabeled_probs, _ = ensemble_predict(model_paths, X_unlabeled)

    print(f"T sources predicted:         {len(t_probs)}")
    print(f"Unlabeled sources predicted: {len(unlabeled_probs)}")

    results = {
        "test_probs":      test_probs,
        "test_preds":      test_preds,
        "y_test":          y_test,
        "test_acc":        test_acc,
        "test_data":       test_data,
        "val_probs":       val_probs,
        "val_preds":       val_preds,
        "y_val":           y_val,
        "t_probs":         t_probs,
        "t_data":          d["t_data"],
        "unlabeled_probs": unlabeled_probs,
        "unlabeled_data":  d["unlabeled_data"],
        "all_test_probs":  all_test_probs,
        "val_accs":        val_accs,
        "model_paths":     model_paths,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved results to {OUTPUT_PATH}")
    print(f"Step 3 complete.")