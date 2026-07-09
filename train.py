"""
Training Script: Cats vs Dogs Classifier
========================================
This script loads the dataset from the `dataset` directory, 
prepares it, builds a classifier using Transfer Learning (MobileNetV2),
trains it, and saves the trained model and class labels.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# --- Configuration ---
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "training_set")
TEST_DIR = os.path.join(DATASET_DIR, "test_set")

BATCH_SIZE = 32
IMAGE_SIZE = (160, 160)  # Use 160x160 to train faster with good accuracy
EPOCHS = 5
MODEL_SAVE_PATH = "models/cats_dogs_model.h5"
CLASSES_SAVE_PATH = "cats_dogs_classes.txt"


def load_datasets():
    """Load train and test datasets from directory."""
    print("⏳ Loading training dataset...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
    )

    print("⏳ Loading validation (test) dataset...")
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        shuffle=False,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
    )

    # Get class names
    class_names = train_dataset.class_names
    print(f"✅ Found classes: {class_names}")

    # Save class names to text file for FastAPI
    with open(CLASSES_SAVE_PATH, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"📝 Saved class labels to: {CLASSES_SAVE_PATH}")

    # Map dataset to normalize pixels from [0, 255] to [0.0, 1.0]
    train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
    validation_dataset = validation_dataset.map(lambda x, y: (x / 255.0, y))

    # Configure dataset performance
    # Prefetch overlays the preprocessing and model execution of a training step
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, class_names


def build_model(num_classes):
    """
    Build a model using Transfer Learning (MobileNetV2).
    Pre-trained weights from ImageNet are frozen, and we add a custom head.
    """
    print("🧠 Building model using MobileNetV2 (Transfer Learning)...")
    
    # 1. Input layer
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # 2. Data augmentation (helps prevent overfitting)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)

    # 3. Preprocessing layer:
    # The model will receive normalized [0.0, 1.0] inputs (matching FastAPI),
    # and we rescale it to [-1.0, 1.0] as expected by MobileNetV2.
    x = layers.Rescaling(2.0, offset=-1.0)(x)

    # 4. Base model (MobileNetV2, pre-trained on ImageNet, without top dense layers)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze the base model weights
    base_model.trainable = False
    
    x = base_model(x, training=False)

    # 5. Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes)(x)  # Raw logits

    # Compile the final model
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    model.summary()
    return model


def train_and_save():
    # Create models directory if not exists
    os.makedirs("models", exist_ok=True)

    # 1. Load data
    train_ds, val_ds, class_names = load_datasets()

    # 2. Build model
    model = build_model(len(class_names))

    # 3. Train
    print(f"🚀 Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 4. Save model
    print(f"💾 Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model saved successfully!")

    # 5. Plot training results
    plot_results(history)


def plot_results(history):
    """Plot accuracy and loss over epochs."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    # Save training graph
    graph_path = "logs/training_history.png"
    plt.savefig(graph_path)
    print(f"📊 Training history plot saved to: {graph_path}")
    plt.close()


if __name__ == "__main__":
    train_and_save()
