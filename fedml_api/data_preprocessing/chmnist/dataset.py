import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
import os
from PIL import Image
from tqdm import tqdm
from pdb import set_trace as st

def load_CH_MNIST():
    """
    Loads CH_MNIST dataset and maps it to Target Model and Shadow Model.
    :param model_mode: one of "TargetModel" and "ShadowModel".
    :return: Tuple of numpy arrays:'(x_train, y_train), (x_test, y_test), member'.
    :raise: ValueError: in case of invalid `model_mode`.
    """

    # Initialize Data
    images, labels = tfds.load('colorectal_histology', split='train', batch_size=-1, as_supervised=True)

    x_train, x_test, y_train, y_test = train_test_split(images.numpy(), labels.numpy(), train_size=0.8,
                                                        random_state=1,
                                                        stratify=labels.numpy())

    x_train = tf.image.resize(x_train, (64, 64))
    y_train = tf.keras.utils.to_categorical(y_train-1, num_classes=8)
    m_train = np.ones(y_train.shape[0])

    x_test = tf.image.resize(x_test, (64, 64))
    y_test = tf.keras.utils.to_categorical(y_test-1, num_classes=8)
    m_test = np.zeros(y_test.shape[0])
    st()
    return (x_train, y_train), (x_test, y_test)

load_CH_MNIST()