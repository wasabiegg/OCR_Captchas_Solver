import sys
import numpy as np

# import matplotlib.pyplot as plt

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import argparse
import logging
from config import BaseConfig
from model import ModelLoader
from utils import encode_single_sample

base_config = BaseConfig.get_config()

logging.basicConfig(format="%(process)d-%(levelname)s-%(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    indices = np.arange(size)

    if shuffle:
        np.random.shuffle(indices)

    train_samples = int(size * train_size)

    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]

    return x_train, y_train, x_valid, y_valid


def train(epochs: int, batch_size: int, data_dir: Path) -> None:
    if not data_dir.is_dir():
        logger.error(f"data dir not found: {str(data_dir)}")
        sys.exit(1)
    images = sorted(list(map(str, list(data_dir.glob("*.png")))))
    labels = [Path(img).stem for img in images]
    characters = list(set(char for label in labels for char in label))
    characters.sort()

    max_length = max([len(label) for label in labels])
    base_config.max_length = max_length

    print("Number of images found: ", len(images))
    print("Number of labels found: ", len(labels))
    print("Number of unique characters: ", len(characters))
    print("Characters present: ", characters)

    # get model
    model_dir = base_config.model_dir
    model = ModelLoader.get_model()

    print(model.summary())

    # load train and val dataset
    x_train, y_train, x_valid, y_valid = split_data(np.array(images), np.array(labels))
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # set up training params
    early_stopping_patience = 10
    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    # save model
    print(f"Saving model: {str(model_dir)}")
    model.save(str(model_dir))

    print("done")


def main() -> None:
    parser = argparse.ArgumentParser(description="training utils")
    parser.add_argument("--epochs", type=int, help="training epochs", default=100)
    parser.add_argument("--batch", type=int, help="batch size", default=16)

    args = parser.parse_args()

    train(args.epochs, args.batch, base_config.data_dir)


if __name__ == "__main__":
    main()
