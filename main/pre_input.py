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

import tarfile
import os


"""
This input file is prepared to read a flower dataset. You can find the dataset in:
http://download.tensorflow.org/example_images/flower_photos.tgz
"""


def unzip_input(compressed_file, dest_path):
    """ Unzip the zip file with all the images from the dataset.

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


def create_datasets(images_path, save_path):
    """ Generates the eval and train datasets from the photos directory. It also creates a file with
        the class names for the labels.

        Args:
            images_path: path to the folder where the input is stored
            save_path: path where the file are being stored
    """
    if not os.path.exists(images_path):
        print('Data path', images_path, "does not exist.")
        return

    print("Creating datasets...")

    dir_paths = _get_directories(images_path)
    class_names = _get_class_names(images_path)

    _create_training_set(dir_paths, os.path.join(save_path, 'training_set.txt'))
    _create_eval_set(dir_paths, os.path.join(save_path, 'eval_set.txt'))
    _create_label_file(class_names, os.path.join(save_path, 'labels.txt'))

    print("All datasets created correctly in", save_path)


def _get_directories(data_path):
    """ Get a list of directories where the photos are stored.

        Args:
            data_path: path to the folder where the folders containing the input are stored

        Return:
            dir_paths: list of the paths where the input is stored
    """
    dir_paths = []

    for item in os.listdir(data_path):
        directory = os.path.join(data_path, item)
        if os.path.isdir(directory):
            dir_paths.append(directory)

    return dir_paths


def _get_class_names(data_path):
    """ Get a list with the class names. This list is get from the folders' name.

        Args:
            data_path: path to the folder where the folders containing the input are stored

        Return:
            class_names: list of the classes' names.
    """
    class_names = []

    for item in os.listdir(data_path):
        directory = os.path.join(data_path, item)
        if os.path.isdir(directory):
            class_names.append(item)

    return class_names


def _create_training_set(data_paths, training_set_path, number_of_items_per_class=500):
    """ Create a text file storing the path to the images and there class label in one row.

        Args:
            data_paths: list of the paths where the input is stored
            training_set_path: path where the file is being stored
            number_of_items_per_class: Number of items per class you want to have in your training dataset. This
                number must be smaller than the maximum number of images per class. By default each class will
                have 500 entries for training.
    """
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
    """ Create a text file storing the path to the images and there class label in one row.

        Args:
            data_paths: list of the paths where the input is stored
            eval_set_path: path where the file is being stored
            number_of_items_per_class: Number of items per class you want to have in your training dataset. This
                number must be smaller than the maximum number of images per class. The images used for creating the
                eval dataset are the ones which are not used in training.
    """
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
    """ Saves the classes' names in a txt file. """
    if os.path.exists(labels_path):
        print("Training set file already exists.")
        return

    labels_file = open(labels_path, 'w')

    for class_name in class_names:
        labels_file.write("%s %s \n" % (class_name, class_names.index(class_name)))

    labels_file.close()
