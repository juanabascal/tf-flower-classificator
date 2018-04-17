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
from datetime import datetime

training_set_path = "/home/uc3m4/PycharmProjects/ft_flowers/data/training_set.txt"


def train():
    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint("/home/uc3m4/PycharmProjects/ft_flowers/data/checkpoints"))

        images, labels, images_placeholder, labels_placeholder, iterator = input.generate_batch_in_iterator("/home/uc3m4/PycharmProjects/ft_flowers/data/training_images.npy",
                                                                                            "/home/uc3m4/PycharmProjects/ft_flowers/data/training_labels.npy", 32)

        next_batch = iterator.get_next()
        next_batch_images, next_batch_labels = next_batch

        # Get the bottleneck tensor
        bottleneck, end_points = model.inception_v4(next_batch_images, num_classes=None)
        logits = model.fine_tuning(bottleneck, end_points)
        loss = tf.reduce_mean(model.loss(logits, labels=tf.one_hot(next_batch_labels, 5)))
        optimizer = tf.train.GradientDescentOptimizer(1)
        train_op = optimizer.minimize(loss)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(iterator.initializer, feed_dict={images_placeholder: images,
                                                      labels_placeholder: labels})

            sess.run(next_batch)
            sess.run(next_batch_labels)


            print('Empieza entrenamiento')
            for i in range(0, 100):
                sess.run(train_op)
                if i % 10 is 0:
                    print('Time:', datetime.now(), 'Loss:', sess.run(loss), 'Step:', i)
                else:
                    print('Time:', datetime.now(), 'Step:', i)
                sess.run(next_batch)
                sess.run(next_batch_labels)
            print('Termina entrenamiento')


def train_function(total_loss):
    optimizer = tf.train.GradientDescentOptimizer(0.9)
    train_op = tf.contrib.slim.learning.create_train_op(total_loss, optimizer)
    logdir = './data/checkpoints/fine-tuning'

    tf.contrib.slim.learning.train(
        train_op,
        logdir,
        number_of_steps=1000,
        save_summaries_secs=300,
        save_interval_secs=600)


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()
