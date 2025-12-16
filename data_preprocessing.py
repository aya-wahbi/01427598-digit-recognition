import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_data(validation_split=0.2):
    """
    Load and preprocess the MNIST dataset.
    :param validation_split: Fraction of the training data to use as validation data.
    :return: (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
    """

    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    # normalize the pixel values to the range [0, 1]
    train_data = train_data.astype("float32") / 255.0
    test_data = test_data.astype("float32") / 255.0

    # reshape data to add channel dimension (which is required for CNN)
    train_data = np.expand_dims(train_data, axis=-1)  # Shape: (60000, 28, 28, 1)
    test_data = np.expand_dims(test_data, axis=-1)    # Shape: (10000, 28, 28, 1)

    #one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=10)
    test_labels = to_categorical(test_labels, num_classes=10)

    # Split training data into train and validation sets
    num_training_samples = int(train_data.shape[0] * (1 - validation_split))
    val_data = train_data[num_training_samples:]
    val_labels = train_labels[num_training_samples:]
    train_data = train_data[:num_training_samples]
    train_labels = train_labels[:num_training_samples]

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)



def augment_data(data, labels, batch_size=32):
    data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    return data_augmentation.flow(data, labels, batch_size=batch_size)