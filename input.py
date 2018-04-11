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
import random
from PIL import Image
import numpy as np

IMAGE_HEIGHT = 299
IMAGE_WIDTH = 299

"""
This input file is prepared to read a flower dataset. You can find the dataset in:
http://download.tensorflow.org/example_images/flower_photos.tgz
"""


def _read_images(training_entry):
    """
    https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html
    https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
    """
    class Record(object):
        pass

    input_tensors = Record()

    file_path, label = _get_image_and_label_from_entry(training_entry)

    input_tensors.image = Image.open(file_path)

    print("Size:", input_tensors.image.size)

    if input_tensors.image.size[0] < IMAGE_HEIGHT or input_tensors.image.size[1] < IMAGE_WIDTH:
        input_tensors.image = input_tensors.image.resize((IMAGE_HEIGHT, IMAGE_WIDTH))

    #input_tensors.image = tf.image.decode_jpeg(image, channels=3)
    input_tensors.label = tf.constant([label], tf.int32)

    return input_tensors.image, input_tensors.label


def _get_image_and_label_from_entry(training_entry):
    file_path, label = training_entry.split(" ")[0:2]

    return file_path, int(label)


def distorted_input_entry(training_entry):
    image, label = _read_images(training_entry)
    with tf.name_scope('data_augmentation'):
        reshaped_image = tf.cast(image, tf.float16)

        # Randomly crop the input image
        distorted_image = tf.random_crop(reshaped_image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

        # Randomly flip the input image
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly modify the brightness and contrast
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

        # Image normalization
        norm_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of the tensors
        norm_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        label.set_shape([1])

    return norm_image, label


def _generate_batch(image, label):
        # TODO: Repasar que hacen todos los argumentos
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=32,
                                                capacity=10000,
                                                min_after_dequeue=3000,
                                                num_threads=16)

        return images, tf.reshape(labels, [32])


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


def get_random_entries(path, batch_size):
    entries = []
    random_entries = []
    for entry in open(path).readlines():
        entries.append(entry)

    for i in range(0, batch_size):
        random_entries.append(entries[random.randint(0, len(entries)-1)])

    return entries


def create_numpy_file(image_set):
    images = []
    labels = []
    i = 0

    for entry in open(image_set).readlines():
        image, label = _get_image_and_label_from_entry(entry)

        image = Image.open(image)

        # Resize the image to have the required dimensions
        image = image.resize((IMAGE_HEIGHT, IMAGE_WIDTH), Image.BILINEAR)
        images.append(np.array(image))

        # Convert the label to int
        label = int(label)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    np.save("/home/uc3m4/PycharmProjects/ft_flowers/data/training_images", arr=images)
    np.save("/home/uc3m4/PycharmProjects/ft_flowers/data/training_labels", arr=labels)

    print("Ficheros npy creados.")


def main(none):
    unzip_input('/home/uc3m4/PycharmProjects/ft_flowers/data/flower_photos.tgz',
                '/home/uc3m4/PycharmProjects/ft_flowers/data/photos')
    prepare_training_dataset('/home/uc3m4/PycharmProjects/ft_flowers/data/photos/flower_photos/')
    create_numpy_file("/home/uc3m4/PycharmProjects/ft_flowers/data/training_set.txt")


if __name__ == "__main__":
    tf.app.run()