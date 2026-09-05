import tensorflow as tf
from pathlib import Path

from dataset import train_dataset, validation_dataset
from model import model


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "efficientnetb0_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "efficientnetb0_final.keras"


# =========================
# Training Configuration
# =========================

EPOCHS = 10


# =========================
# Callbacks
# =========================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7
    )
]


# =========================
# Training
# =========================

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================
# Save Final Model
# =========================

model.save(FINAL_MODEL_PATH)

print("\nTraining completed.")
print(f"Best model saved to: {BEST_MODEL_PATH}")
print(f"Final model saved to: {FINAL_MODEL_PATH}")