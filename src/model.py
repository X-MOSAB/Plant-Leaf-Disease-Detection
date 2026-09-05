import tensorflow as tf

# =========================
# Configuration
# =========================

IMG_SIZE = (224, 224)
NUM_CLASSES = 39


# =========================
# Data Augmentation
# =========================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")


# =========================
# Build EfficientNetB0 Model
# =========================

def build_model(num_classes=NUM_CLASSES):

    # Pretrained base model
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3)
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))

    x = data_augmentation(inputs)

    # EfficientNetB0 includes its own input preprocessing
    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    return model


# =========================
# Create Model
# =========================

model = build_model()

# =========================
# Compile Model
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# Model Summary
# =========================

model.summary()