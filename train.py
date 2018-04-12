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
from model import inception_v4
import input

training_set_path = "/home/uc3m4/PycharmProjects/ft_flowers/data/training_set.txt"


def train():
    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint("/home/uc3m4/PycharmProjects/ft_flowers/data/checkpoints"))

            for step in range(0, 20):
                entries = input.get_random_entries(training_set_path, 32)

                for entry in entries:
                    image, label = input.distorted_input_entry(entry)
                    logits, end_points = inception_v4(tf.expand_dims(image, 0))

                    # Set up all our weights to their initial default values.
                    init = tf.global_variables_initializer()
                    sess.run(init)

                    print(sess.run(logits))

                print("Step", step)


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()
