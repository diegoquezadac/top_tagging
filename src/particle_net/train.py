import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import tensorflow as tf
from tensorflow import keras
from src.particle_net.dataset import PointDataset, create_data_loader
from src.particle_net.model import get_particle_net


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 20:
        lr *= 0.01
    elif epoch > 10:
        lr *= 0.1
    return lr

class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, name="weighted_categorical_crossentropy"):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)

    def call(self, y_true, y_pred, sample_weight=None):
        # Compute categorical crossentropy loss without reduction
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        tf.print("Loss shape before weighting:", tf.shape(loss))  # Debug shape
        if sample_weight is not None:
            tf.print("Sample weights in loss:", sample_weight)
            loss = loss * sample_weight
        else:
            tf.print("No sample weights provided")
        # Manually reduce with mean
        return tf.reduce_mean(loss)

if __name__ == "__main__":

    # Define variables
    batch_size = 10
    epochs = 30
    lr = 4.2e-4
    val_split = 0.2

    train_ds = PointDataset("./data/test-preprocessed.h5")
    data_loader = create_data_loader(train_ds, batch_size=batch_size)

    num_classes = 2 # train_ds.y.shape[1]
    input_shapes = {"features": (80, 7), "points": (80, 2)} # {k: train_ds[k].shape[1:] for k in train_ds.X}

    model = get_particle_net(num_classes, input_shapes)
    
    save_dir = "checkpoints/particle_net"
    model_name = "best_model.h5"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, model_name)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    progress_bar = keras.callbacks.ProgbarLogger()
    callbacks = [checkpoint, lr_scheduler, progress_bar]

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = WeightedCategoricalCrossentropy()

    # Single training step
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        total_loss = 0.0
        num_batches = 0
        total_samples = 0

        # Iterate over all batches in the dataset until exhausted
        for batch in data_loader:
            inputs, labels, weights = batch
            batch_size_actual = tf.shape(labels)[0].numpy()  # Handle partial batches
            total_samples += batch_size_actual
            print(f"Batch {num_batches + 1} - Batch weights:", weights.numpy())

            # Training step
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn.call(labels, predictions, sample_weight=weights)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Update metrics
            total_loss += loss.numpy()
            num_batches += 1

            # Print progress every 10 batches
            if (num_batches + 1) % 10 == 0:
                print(f"Batch {num_batches + 1}, Loss: {loss.numpy():.4f}")

        # Print epoch summary
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}, Samples processed: {total_samples}")