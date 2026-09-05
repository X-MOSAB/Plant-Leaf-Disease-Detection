from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from dataset import test_dataset, class_names


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "efficientnetb0_best.keras"


# =========================
# Load Model
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# =========================
# Test Evaluation
# =========================

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\n" + "=" * 50)
print("TEST RESULTS")
print("=" * 50)

print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4%}")


# =========================
# Predictions on Test Set
# =========================

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


# =========================
# Classification Report
# =========================

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

print("\nConfusion Matrix Shape:", cm.shape)



# =========================
# Save Confusion Matrix
# =========================

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

plt.figure(figsize=(20, 18))

plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - EfficientNetB0")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(len(class_names)),
    class_names,
    fontsize=7
)

plt.colorbar()

plt.tight_layout()

confusion_matrix_path = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion matrix saved to: {confusion_matrix_path}"
)