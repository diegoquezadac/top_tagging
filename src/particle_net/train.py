import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from src.particle_net.dataset import PointDataset, random_split
from tensorflow import keras
from src.particle_net.model import get_particle_net


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 20:
        lr *= 0.01
    elif epoch > 10:
        lr *= 0.1
    return lr


if __name__ == "__main__":

    # Define variables
    max_jets = 500
    batch_size = 250
    epochs = 30
    lr = 4.2e-4
    val_split = 0.2

    # Define data sets
    dataset = PointDataset("./data/test-public-small.h5")
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Define model
    model_type = "particle_net"
    num_classes = train_ds.y.shape[1]
    input_shapes = {k: train_ds[k].shape[1:] for k in train_ds.X}
    model = get_particle_net(num_classes, input_shapes)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
        metrics=["accuracy"],
    )

    # Train model
    save_dir = "model_checkpoints"
    model_name = f"{model_type}_model.{{epoch:03d}}.h5"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, model_name)

    # ✅ correct monitor name
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

    history = model.fit(
        train_ds.X,
        train_ds.y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_ds.X, val_ds.y),
        shuffle=True,
        callbacks=callbacks,
    )
