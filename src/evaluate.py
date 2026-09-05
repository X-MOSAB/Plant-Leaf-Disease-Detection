from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from dataset import test_dataset, class_names


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "efficientnetb0_best.keras"


# =========================
# Load Best Model
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")


# =========================
# Evaluate on Test Set
# =========================

test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

print("\n" + "=" * 50)
print("TEST RESULTS")
print("=" * 50)

print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4%}")


# =========================
# Predictions
# =========================

y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)

    predicted_classes = tf.argmax(
        predictions,
        axis=1
    ).numpy()

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

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix Shape:", cm.shape)