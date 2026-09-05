from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

from dataset import class_names


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "efficientnetb0_best.keras"

IMG_SIZE = (224, 224)


# =========================
# Load Model
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# =========================
# Prediction Function
# =========================

def predict_image(image_path: str):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(predictions)[::-1][:3]

    return [
        (class_names[i], float(predictions[i]))
        for i in top_indices
    ]


# =========================
# Main
# =========================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("\nUsage:")
        print("python src/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    results = predict_image(image_path)

    print("\n" + "=" * 50)
    print("TOP 3 PREDICTIONS")
    print("=" * 50)

    for rank, (class_name, confidence) in enumerate(results, start=1):
        print(
            f"{rank}. {class_name}: {confidence:.2%}"
        )