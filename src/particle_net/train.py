import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.particle_net.dataset import PointDataset, create_data_loader
from src.particle_net.model import get_particle_net
from src.utils import get_logger

logger = get_logger("particle-net_training")

def lr_schedule(epoch):
    lr = 4.2e-4
    if epoch > 20:
        lr *= 0.01
    elif epoch > 10:
        lr *= 0.1
    return lr

class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, name="weighted_categorical_crossentropy"):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)
    def call(self, y_true, y_pred, sample_weight=None):
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        if sample_weight is not None:
            loss = loss * sample_weight
        return tf.reduce_mean(loss)

def compute_br(y_true, y_pred, sample_weight, target_eff=0.8):
    signal_labels = y_true[:, 0]
    signal_scores = y_pred[:, 0]
    sample_weight = sample_weight if sample_weight is not None else np.ones_like(signal_labels)

    # Find threshold for target signal efficiency (TPR)
    sorted_indices = np.argsort(signal_scores)[::-1]
    sorted_scores = signal_scores[sorted_indices]
    sorted_labels = signal_labels[sorted_indices]
    sorted_weights = sample_weight[sorted_indices]
    
    cumsum_signal = np.cumsum(sorted_weights * (sorted_labels == 1))
    total_signal = np.sum(sorted_weights * (sorted_labels == 1)) + 1e-10
    tpr_values = cumsum_signal / total_signal
    
    idx = np.searchsorted(tpr_values, target_eff, side='left')
    if idx >= len(tpr_values):
        idx = len(tpr_values) - 1
    threshold = sorted_scores[idx]

    # Compute FPR at this threshold
    predictions = (signal_scores > threshold).astype(np.float32)
    background_mask = signal_labels == 0
    fp = np.sum(sample_weight[background_mask & (predictions == 1)])
    tn = np.sum(sample_weight[background_mask & (predictions == 0)])
    fpr = fp / (fp + tn + 1e-10)
    
    return 1 / fpr, threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ParticleNet model training")
    parser.add_argument(
        "input_path",
        type=str,
        default="./data/test-preprocessed.h5",
        help="Path to the training file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the best_model.h5 checkpoint",
    )
    args = parser.parse_args()

    batch_size = 64
    num_epochs = 30
    val_split = 0.1
    target_eff = 0.8
    max_jets = 4000000
    max_constits = 80
    checkpoint_dir = Path("checkpoints/particle_net")
    checkpoint_path = checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataset
    full_dataset = PointDataset(args.input_path, max_constits=max_constits, max_jets=max_jets)
    full_dataset.inspect_hdf5()
    num_samples = len(full_dataset)

    # Split indices
    indices = np.random.permutation(num_samples)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    logger.info(f"Train dataset: {train_size} data points")
    logger.info(f"Validation dataset: {val_size} data points")

    train_ds = PointDataset(args.input_path, indices=train_indices.tolist(), max_constits=max_constits)
    val_ds = PointDataset(args.input_path, indices=val_indices.tolist(), max_constits=max_constits)

    train_loader = create_data_loader(train_ds, batch_size=batch_size)
    val_loader = create_data_loader(val_ds, batch_size=batch_size)

    num_classes = 2
    input_shapes = {"features": (80, 7), "points": (80, 2)}
    model = get_particle_net(num_classes, input_shapes)
    logger.info(f"Total trainable parameters: {model.count_params()}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule(0))
    loss_fn = WeightedCategoricalCrossentropy()

    history = {"train_loss": [], "val_loss": [], "train_br": [], "val_br": []}
    best_val_loss = float("inf")
    start_epoch = 1

    if args.resume and checkpoint_path.exists():
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint.restore(latest).expect_partial()
        # Load history and metadata
        checkpoint_data = np.load(checkpoint_dir / "checkpoint_data.npz", allow_pickle=True)
        history = checkpoint_data["history"].item()
        best_val_loss = checkpoint_data["best_val_loss"]
        start_epoch = int(checkpoint_data["epoch"]) + 1
        saved_max_constits = checkpoint_data["max_constits"]
        saved_val_split = checkpoint_data["val_split"]
        saved_input_path = checkpoint_data["input_path"]
        if saved_max_constits != max_constits or saved_val_split != val_split or saved_input_path != args.input_path:
            logger.info("Warning: Dataset parameters have changed since checkpoint!")
        logger.info(f"Resuming training from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
    elif args.resume:
        logger.info(f"Error: Checkpoint file {checkpoint_path} not found!")
        sys.exit(1)

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        optimizer.learning_rate.assign(lr_schedule(epoch))
        logger.info(f"Learning rate: {optimizer.learning_rate.numpy():.6f}")

        total_loss = 0.0
        num_batches = 0
        total_samples = 0
        train_labels = []
        train_preds = []
        train_weights = []

        for batch in train_loader:
            inputs, labels, weights = batch
            batch_size_actual = tf.shape(labels)[0].numpy()
            total_samples += batch_size_actual

            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn.call(labels, predictions, sample_weight=weights)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_labels.append(labels.numpy())
            train_preds.append(predictions.numpy())
            train_weights.append(weights.numpy() if weights is not None else np.ones(batch_size_actual))

            total_loss += loss.numpy()
            num_batches += 1

            if (num_batches + 1) % 10 == 0:
                logger.info(f"Batch {num_batches + 1}, Loss: {loss.numpy():.4f}")

        train_labels = np.concatenate(train_labels, axis=0)
        train_preds = np.concatenate(train_preds, axis=0)
        train_weights = np.concatenate(train_weights, axis=0)
        train_br, train_threshold = compute_br(train_labels, train_preds, train_weights, target_eff=target_eff)

        val_loss = 0.0
        val_batches = 0
        val_samples = 0
        val_labels = []
        val_preds = []
        val_weights = []

        for batch in val_loader:
            inputs, labels, weights = batch
            batch_size_actual = tf.shape(labels)[0].numpy()
            val_samples += batch_size_actual
            predictions = model(inputs, training=False)
            loss = loss_fn.call(labels, predictions, sample_weight=weights)

            val_labels.append(labels.numpy())
            val_preds.append(predictions.numpy())
            val_weights.append(weights.numpy() if weights is not None else np.ones(batch_size_actual))

            val_loss += loss.numpy()
            val_batches += 1

        if not val_labels:
            logger.info("Validation loader is empty!")
            continue

        val_labels = np.concatenate(val_labels, axis=0)
        val_preds = np.concatenate(val_preds, axis=0)
        val_weights = np.concatenate(val_weights, axis=0)
        val_br, val_threshold = compute_br(val_labels, val_preds, val_weights, target_eff=target_eff)

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_br"].append(train_br)
        history["val_br"].append(val_br)

        logger.info(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Train BR: {train_br:.4f} (Threshold: {train_threshold:.4f}), "
              f"Val Loss: {avg_val_loss:.4f}, Val BR: {val_br:.4f} (Threshold: {val_threshold:.4f}), "
              f"Train Samples: {total_samples}, Val Samples: {val_samples}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
            checkpoint.save(file_prefix=checkpoint_dir / "ckpt")
            np.savez(
                checkpoint_dir / "checkpoint_data.npz",
                epoch=epoch,
                best_val_loss=best_val_loss,
                history=history,
                max_constits=max_constits,
                val_split=val_split,
                input_path=args.input_path,
                lr=optimizer.learning_rate.numpy()
            )
            logger.info(f"Saved checkpoint with Val Loss: {best_val_loss:.4f}")

        # Plot loss curves
        plt.plot(history["train_loss"], label="Training")
        plt.plot(history["val_loss"], label="Validation")
        plt.ylabel("Cross-entropy Loss")
        plt.xlabel("Training Epoch")
        plt.ylim(0, 1)
        plt.legend()
        figure_dir = Path("figures")
        figure_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_dir / "loss_particle_net.png", dpi=300)
        plt.clf()