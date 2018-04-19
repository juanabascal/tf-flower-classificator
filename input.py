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


def _get_image_and_label_from_entry(dataset_entry):
    file_path, label = dataset_entry.split(" ")[0:2]

    return file_path, int(label)


def main(none):
    """ Run this function to create the datasets and the numpy array files. """

    pre_input.unzip_input(FLAGS.zip_file_path, os.path.join(FLAGS.data_path, "images"))
    pre_input.create_datasets(FLAGS.images_path, FLAGS.data_path)
    generate_tfrecord_files(os.path.join(FLAGS.data_path, "training_set.txt"),
                            os.path.join(FLAGS.data_path, "flowers.tfrecord"))
    consume_tfrecord()


if __name__ == "__main__":
    tf.app.run()
