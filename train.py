# Copyright 2018 Juan Abascal & Daniel Gonzalez. All Rights Reserved.
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
import logging
import input
from datetime import datetime

tf.app.flags.DEFINE_string('ckpt_dir', '/home/uc3m4/PycharmProjects/ft_flowers/data/checkpoint',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Size of batches.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

FLAGS = tf.app.flags.FLAGS


def train():

    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()

        iterator = input.consume_tfrecord()
        images_batch, labels_batch = iterator.get_next()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint("./data/checkpoints"))

        # Num_classes is None for fine tunning
        bottleneck, end_points = model.inception_v4(images_batch, num_classes=None, is_training=False)
        logits = model.fine_tuning(bottleneck, end_points)

        # TODO: Add a function to get train_op
        loss = model.loss(logits, labels_batch)
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        train_op = optimizer.minimize(loss, global_step=global_step)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run([images_batch, labels_batch])

            logger = init_logger()
            logger.info("Training starts...")
            for i in range(0, 2000):
                sess.run(train_op)
                if i % 10 is 0:
                    print("ahi va")
                    logger.info('Time: %s Loss: %f Step: %i', datetime.now(), sess.run(loss), i)
                else:
                    logger.info('Time: %s Step: %i', datetime.now(), i)

            logger.info("Training ends...")


def main(argv=None):
    train()


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


if __name__ == "__main__":
    tf.app.run()
