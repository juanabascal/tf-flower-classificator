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
from PIL import Image
import numpy as np

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


def create_numpy_file(image_set, save_path):
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

    np.save(os.path.join(save_path, "training_images"), arr=images)
    np.save(os.path.join(save_path, "training_labels"), arr=labels)

    print("Npy files created correctly in", images_path, "and", labels_path)


def generate_batch_in_iterator(npy_images_file, npy_labels_file, batch_size=1):
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
    unzip_input(FLAGS.zip_file_path, os.path.join(FLAGS.data_path, "images"))
    create_datasets(FLAGS.images_path, FLAGS.data_path)
    create_numpy_file(os.path.join(FLAGS.data_path, "training_set.txt"), FLAGS.data_path)


if __name__ == "__main__":
    tf.app.run()
