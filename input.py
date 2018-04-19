# Copyright 2018 Juan Abascal. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pre_input
import os
from PIL import Image
import numpy as np
import tfrecord_utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 299,
                            """Height and width of the images.""")
tf.app.flags.DEFINE_string('zip_file_path', "./data/flower_photos.tgz",
                           """Path to the zip file.""")
tf.app.flags.DEFINE_string('data_path', "./data",
                           """Path to the data.""")
tf.app.flags.DEFINE_string('images_path', "./data/images/flower_photos",
                           """Path to the photos.""")


"""
This input file is prepared to read a flower dataset. You can find the dataset in:
http://download.tensorflow.org/example_images/flower_photos.tgz
"""


def convert_to_numpy(image_set, save_path, to_file=False):
    """ TODO: Preparar el dataset para imagenes de cualquier tamaño
    See guide: https://www.tensorflow.org/programmers_guide/datasets#batching_tensors_with_padding
    From the txt datasets 'image_path label' create numpy binary files

        Args:
            image_set: path to the txt with the dataset.
            save_path: the path where you want to save your npy files.
    """
    images_path = os.path.join(save_path, "training_images.npy")
    labels_path = os.path.join(save_path, "training_labels.npy")

    if os.path.exists(images_path) and os.path.exists(labels_path):
        print("Npy files already exists.")
        return

    print("Creating numpy files...")

    images = []
    labels = []

    for entry in open(image_set).readlines():
        image, label = _get_image_and_label_from_entry(entry)

        image = Image.open(image)

        # Resize the image to have the required dimensions
        image = image.resize((FLAGS.image_size, FLAGS.image_size), Image.BILINEAR)
        images.append(np.array(image))

        # Convert the label to int
        label = int(label)
        labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    if not to_file:
        return images, labels

    np.save(os.path.join(save_path, "training_images"), arr=images)
    np.save(os.path.join(save_path, "training_labels"), arr=labels)

    print("Npy files created correctly in", images_path, "and", labels_path)


def distorted_input(image, label):
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image)

    # TODO: Make the order of following operations random.
    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    norm_image = tf.image.per_image_standardization(distorted_image)

    return norm_image, label


def norm_input(image, label):
    norm_image = tf.image.per_image_standardization(image)

    return norm_image, label


def generate_tfrecord_files(image_set, save_file):
    if os.path.exists(save_file):
        print("TFRecord file already exists.")
        return

    print("Creating TFRecord file...")

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(save_file) as writer:

        for entry in open(image_set):
            tf_example = create_tf_example(entry)
            writer.write(tf_example.SerializeToString())

        writer.close()


def create_tf_example(entry):
    image_path, label = _get_image_and_label_from_entry(entry)

    image = Image.open(image_path)
    image_np = np.array(image)
    image_raw = image_np.tostring()

    feature = {
        'image': tfrecord_utils.bytes_feature(image_raw),
        'image/height': tfrecord_utils.int64_feature(image_np.shape[0]),
        'image/width': tfrecord_utils.int64_feature(image_np.shape[1]),
        'label': tfrecord_utils.int64_feature(label),
    }

    tf_label_and_data = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_label_and_data


def consume_tfrecord(distorted=True):
    dataset = tf.data.TFRecordDataset(os.path.join(FLAGS.data_path, "flowers.tfrecord"))
    dataset = dataset.map(tfrecord_utils.parse)

    if distorted is True:
        dataset = dataset.map(distorted_input)
    else:
        dataset = dataset.map(norm_input)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        print(sess.run(next_element))
        print(sess.run(next_element))
        print(sess.run(next_element))


def generate_batch_from_np(npy_images_file, npy_labels_file, batch_size=1):
    """ Load the npy files and creates a dataset and iterator. For more information about tf.Data API:
        https://www.tensorflow.org/programmers_guide/datasets.

        Args:
            npy_images_file: numpy binary file that has the images.
            npy_labels_file: numpy binary file taht has the labels.
            batch_size: number of images and labels in the batch. By default it is 1.

        Return:
            images: Numpy array containing all the images.
            labels: Numpy array containing all the labels.
            iterator: Iterator object for feeding the model.
    """
    images = np.load(npy_images_file)
    labels = np.load(npy_labels_file)

    images_placeholder = tf.placeholder(images.dtype, (None, 299, 299, 3))
    labels_placeholder = tf.placeholder(labels.dtype, None)

    dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))

    # Shuffle the dataset and create a batch
    dataset = dataset.shuffle(buffer_size=2500)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_initializable_iterator()

    # Uncomment this line to try the iterator before feeding the model
    # exec_iter(images, labels, images_placeholder, labels_placeholder, iterator)

    return images, labels, images_placeholder, labels_placeholder, iterator


def _exec_iter(images, labels, images_placeholder, labels_placeholder, iterator):
    """ Just a class for debugging purposes. """
    next_iter = iterator.get_next()

    print(labels)

    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={images_placeholder: images,
                                                  labels_placeholder: labels})
        print(sess.run(next_iter))


def _get_image_and_label_from_entry(dataset_entry):
    file_path, label = dataset_entry.split(" ")[0:2]

    return file_path, int(label)


def main(none):
    """ Run this function to create the datasets and the numpy array files. """
    pre_input.unzip_input(FLAGS.zip_file_path, os.path.join(FLAGS.data_path, "images"))
    pre_input.create_datasets(FLAGS.images_path, FLAGS.data_path)
    convert_to_numpy(os.path.join(FLAGS.data_path, "training_set.txt"), FLAGS.data_path, to_file=True)
    generate_tfrecord_files(os.path.join(FLAGS.data_path, "training_set.txt"),
                            os.path.join(FLAGS.data_path, "flowers.tfrecord"))

    consume_tfrecord()


if __name__ == "__main__":
    tf.app.run()
