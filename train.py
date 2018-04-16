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
import model
import input

training_set_path = "/home/uc3m4/PycharmProjects/ft_flowers/data/training_set.txt"


def train():
    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint("/home/uc3m4/PycharmProjects/ft_flowers/data/checkpoints"))

        images, labels, images_placeholder, labels_placeholder, iterator = input.generate_batch_in_iterator("/home/uc3m4/PycharmProjects/ft_flowers/data/training_images.npy",
                                                                                            "/home/uc3m4/PycharmProjects/ft_flowers/data/training_labels.npy", 32)

        next_element = iterator.get_next()

        # Get the bottleneck tensor
        bottleneck, end_points = model.inception_v4(next_element[0], num_classes=None)
        logits = model.fine_tuning(bottleneck, end_points)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(iterator.initializer, feed_dict={images_placeholder: images,
                                                      labels_placeholder: labels})

            for i in range(0, 1):
                sess.run(next_element)
                print(sess.run(logits))
                print("Step:", i)


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()
