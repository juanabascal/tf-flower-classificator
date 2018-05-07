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

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# _bytes is used for string/char values
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def parse(serialized):
    """ Convert the images and labels from records feature to Tensors.

        Args:
            serialized: A dataset comprising records from one TFRecord file.

    """

    # Define a dict with the data-names and types we expect to find in the TFRecords file.
    feature = {
        'image': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=feature)

    # Get the image as raw bytes, and  the height, width and label as int.
    image_raw = parsed_example['image']
    height = tf.cast(parsed_example['image/height'], tf.int32)
    width = tf.cast(parsed_example['image/width'], tf.int32)
    label = tf.cast(parsed_example['label'], tf.int32)

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.decode_raw(image_raw, tf.uint8)

    # The type is now uint8 but we need it to be float.
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, (height, width, 3))

    # The image and label are now correct TensorFlow types.
    return image, label
