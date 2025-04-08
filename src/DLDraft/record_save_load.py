import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from tensorflow import data as tf_data
from tensorboard.plugins.hparams import api as hp

from sklearn.model_selection import train_test_split

from preprocess_common import *

AUTO = tf_data.AUTOTUNE

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value, quality=100).numpy()])
    )
    
def label_feature(value):
    """Returns a int64_list from a one hot encoded label."""
    value = int(tf.argmax(value).numpy())
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example(image, label):
    """Create a tf.train.Example from an image and label."""
    feature = {
        "image": image_feature(image),
        "label": label_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_tfrecord_fn(example, model):
    """Parse the serialized example.
    
    Args:
        example (): example
        model (str): name of the model
        Returns:
        (): image and label
    """
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = apply_model_specific_preprocessing(image, model)
    label = keras.ops.one_hot(example["label"], 2)
    return image, label

def save_to_tfrecord(dataset, filename):
    """Save the dataset to a TFRecord file.
    
    Args:
        dataset (): dataset
        filename (str): name of the TFRecord file
    """
    with tf.io.TFRecordWriter(filename) as writer:
        for image_batch, label_batch in dataset:
            for i in range(image_batch.shape[0]):  # Iterate over batch elements
                example = create_example(image_batch[i], label_batch[i])
                writer.write(example.SerializeToString())
                
def load_tfrecord(filename, batch_size, model="efficientnet"):
    """Load a TFRecord file.
    
    Args:
        filename (str): name of the TFRecord file
        model (str): name of the model
        Returns:
        (): dataset
    """
    dataset = tf_data.TFRecordDataset(filename)
    dataset = dataset.map(lambda example: parse_tfrecord_fn(example, model), num_parallel_calls=AUTO, deterministic=True).batch(batch_size, num_parallel_calls=AUTO, deterministic=True).prefetch(AUTO)
    return dataset


def create_test_example(image):
    """Create a tf.train.Example from an image."""
    feature = {
        "image": image_feature(image)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def save_test_tfrecord(dataset, filename):
    """Save the test dataset to a TFRecord file."""
    with tf.io.TFRecordWriter(filename) as writer:
        for image_batch in dataset:
            for i in range(image_batch.shape[0]):  # Iterate over batch elements
                example = create_test_example(image_batch[i])
                writer.write(example.SerializeToString())
                
def parse_test_tfrecord(example, model):
    """Parse the serialized example."""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = apply_model_specific_preprocessing(image, model)
    return image

def load_test_tfrecord(filename, batch_size, model="efficientnet"):
    """Load a test TFRecord file."""
    dataset = tf_data.TFRecordDataset(filename)
    dataset = dataset.map(lambda example: parse_test_tfrecord(example, model), num_parallel_calls=AUTO, deterministic=True).batch(batch_size, num_parallel_calls=AUTO, deterministic=True).prefetch(AUTO)
    return dataset