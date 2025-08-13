import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils import load_from_files
from src.particle_net.dataset import Dataset
from tensorflow import keras
from src.particle_net.model import get_particle_net


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr *= 0.1
    elif epoch > 20:
        lr *= 0.01
    return lr


if __name__ == "__main__":
    # TODO: Split the dataset into train and val instead of creating two datasets

    # Define variables
    max_jets = 500
    batch_size = 250
    epochs = 30
    lr = 4.2e-4

    # Load raw data
    data, labels, _, _, _ = load_from_files(
        ["./data/test-public-small.h5"], max_jets=max_jets, use_train_weights=False
    )

    # Define data sets
    train_dataset = Dataset(data, labels)
    val_dataset = Dataset(data, labels)

    # Define model
    model_type = "particle_net"
    num_classes = train_dataset.y.shape[1]
    input_shapes = {k: train_dataset[k].shape[1:] for k in train_dataset.X}
    model = get_particle_net(num_classes, input_shapes)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
        metrics=["accuracy"],
    )

    # Train model
    save_dir = 'model_checkpoints'
    model_name = '%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                monitor='val_acc',
                                mode='max',
                                verbose=1,
                                save_best_only=True)

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    progress_bar = keras.callbacks.ProgbarLogger()
    callbacks = [checkpoint, lr_scheduler, progress_bar]

    model.fit(
        train_dataset.X,
        train_dataset.y,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_dataset.X, val_dataset.y),
        shuffle=True,
        callbacks=callbacks,
    )