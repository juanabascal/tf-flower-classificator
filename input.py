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
import tarfile
import os

"""
This input file is prepared to read a flower dataset. You can find the dataset in:
http://download.tensorflow.org/example_images/flower_photos.tgz
"""

def _read_images(training_entry):
    """
    https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
    https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
    """
    file_path, label = training_entry.split(" ")

    image_tensor = tf.image.decode_jpeg(file_path)

    pass


class _ImageDecoder(object):
    """ Helper class that provides TensorFlow image coding utilities. """
    def __init__(self):
        self._sess = tf.Session()

        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def _decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3

        return image


def _image_decoder():
    pass


def create_tf_records(file_set):
    with open(file_set, 'r') as file:
        for line in file.readlines():
            _read_images(line.rstrip(' \n'))


def unzip_input(compressed_file, dest_path):
    """ Unzip the file with all the images of the dataset.

    Args:
        compressed_file: the file to be unzipped
        dest_path: path where you want to save the pictures
    """
    if os.path.exists(dest_path):
        print('Destination path', dest_path, "already exists.")
        return

    print('Extracting zip content...')
    zip_ref = tarfile.open(compressed_file, 'r')
    zip_ref.extractall(dest_path)
    zip_ref.close()
    print('All zip content extracted to', dest_path)


def prepare_training_dataset(data_path):
    if not os.path.exists(data_path):
        print('Data path', data_path, "does not exist.")
        return

    dir_paths = _get_directories(data_path)
    class_names = _get_class_names(data_path)

    _create_training_set(dir_paths, '/home/uc3m4/PycharmProjects/ft_flowers/data/training_set.txt')
    _create_label_file(class_names, '/home/uc3m4/PycharmProjects/ft_flowers/data/labels.txt')
    _create_eval_set(dir_paths, '/home/uc3m4/PycharmProjects/ft_flowers/data/eval_set.txt')


def _get_directories(data_path):
    dir_names = []

    for item in os.listdir(data_path):
        directory = os.path.join(data_path, item)
        if os.path.isdir(directory):
            dir_names.append(directory)

    return dir_names


def _get_class_names(data_path):
    class_names = []

    for item in os.listdir(data_path):
        directory = os.path.join(data_path, item)
        if os.path.isdir(directory):
            class_names.append(item)

    return class_names


def _create_training_set(data_paths, training_set_path, number_of_items_per_class=500):

    if os.path.exists(training_set_path):
        print("Training set file already exists.")
        return

    training_set_file = open(training_set_path, 'w')

    for path in data_paths:
        for photo in os.listdir(path)[0:number_of_items_per_class]:
            photo_path = os.path.join(path, photo)
            training_entry = "%s %s \n" % (photo_path, data_paths.index(path))
            training_set_file.write(training_entry)

    training_set_file.close()


def _create_eval_set(data_paths, eval_set_path, number_of_items_per_class=500):
    if os.path.exists(eval_set_path):
        print("Eval set file already exists.")
        return

    eval_set_file = open(eval_set_path, 'w')

    for path in data_paths:
        for photo in os.listdir(path)[number_of_items_per_class:]:
            photo_path = os.path.join(path, photo)
            eval_entry = "%s %s \n" % (photo_path, data_paths.index(path))
            eval_set_file.write(eval_entry)

    eval_set_file.close()


def _create_label_file(class_names, labels_path):
    if os.path.exists(labels_path):
        print("Training set file already exists.")
        return

    labels_file = open(labels_path, 'w')

    for class_name in class_names:
        labels_file.write("%s %s \n" % (class_name, class_names.index(class_name)))

    labels_file.close()


def main(none):
    unzip_input('/home/uc3m4/PycharmProjects/ft_flowers/data/flower_photos.tgz',
                '/home/uc3m4/PycharmProjects/ft_flowers/data/photos')
    prepare_training_dataset('/home/uc3m4/PycharmProjects/ft_flowers/data/photos/flower_photos/')
    create_tf_records('/home/uc3m4/PycharmProjects/ft_flowers/data/training_set.txt')


if __name__ == "__main__":
    tf.app.run()