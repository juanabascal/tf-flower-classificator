# Copyright 2018 by BQ. All Rights Reserved.
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
import os
import tfrecord_utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 299,
                            """Height and width of the images.""")
tf.app.flags.DEFINE_string('data_path', "data",
                           """Path to the data.""")

tfrecord_file_training = os.path.join(FLAGS.data_path, "flowers_train.tfrecord")
tfrecord_file_eval = os.path.join(FLAGS.data_path, "flowers_eval.tfrecord")


"""
This input file is prepared to read a flower dataset. You can find the dataset in:
http://download.tensorflow.org/example_images/flower_photos.tgz
"""


def distorted_input(image, label):
    """ Apply random distortions to the image in order to do data augmentation.

        Args:
            image: the image to apply the distortions.
            label: the label of the image.
        Returns:
            norm_image: the distorted normalized image.
            label: the label is required because the dataset.map function needs both
                elements.
    """
    # Random crop image
    cropped_image = tf.image.resize_image_with_crop_or_pad(image, 324, 324)
    cropped_image = tf.random_crop(cropped_image, [FLAGS.image_size, FLAGS.image_size, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(cropped_image)

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
    """ Apply normalization to the image.

            Args:
                image: the image to apply the normalization.
                label: the label of the image.
            Returns:
                norm_image: the normalized image.
                label: the label is required because the dataset.map function needs both
                    elements.
        """
    cropped_image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.image_size, FLAGS.image_size)

    norm_image = tf.image.per_image_standardization(cropped_image)

    return norm_image, label


def consume_tfrecord(is_training=True, batch_size=32):
    """ Creates a one shot iterator from the TFRecord files.

        Args:
            is_training: if True apply distotion to the images and take the training
                dataset instead of the eval dataset.
            batch_size: size of the batch.
        Return:
             iterator: one shot iterator.
    """
    if is_training:
        dataset = tf.data.TFRecordDataset(tfrecord_file_training)
    else:
        dataset = tf.data.TFRecordDataset(tfrecord_file_eval)

    dataset = dataset.map(tfrecord_utils.parse)

    if is_training:
        dataset = dataset.map(distorted_input)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=2560)
    else:
        dataset = dataset.map(norm_input)

    dataset = dataset.padded_batch(batch_size, padded_shapes=([FLAGS.image_size, FLAGS.image_size, 3], []))

    iterator = dataset.make_one_shot_iterator()

    return iterator
