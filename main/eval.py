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
import logging
from main import input, model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_dir', './data/checkpoints/',
                           """Directory where to restore a model""")
tf.app.flags.DEFINE_string('save_dir', './data/train/flowers',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('log_dir', './data/train/log',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('max_steps', 2000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Size of batches.""")


def train():

    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()

        iterator = input.consume_tfrecord(is_training=False, batch_size=1)
        images_batch, labels_batch = iterator.get_next()

        # Num_classes is None for fine tunning
        with tf.contrib.slim.arg_scope(model.inception_v3_arg_scope()):
            bottleneck, end_points = model.inception_v3(images_batch, num_classes=None, is_training=False)

        # with tf.variable_scope('fine_tuning'):
        logits = model.fine_tuning(bottleneck, end_points)

        saver = tf.train.Saver(tf.global_variables('InceptionV3'))
        saver_ft = tf.train.Saver(tf.global_variables('fine_tuning'))

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            saver_ft.restore(sess, tf.train.latest_checkpoint('/home/uc3m3/Documentos/Trained/ft_flowers/lr_ed_0.2/'))

            tf.summary.image(tensor=images_batch, name="Image")

            logger = init_logger()
            logger.info("Eval starts...")

            correct = 0

            for i in range(1, 1170):
                images, labels = sess.run([images_batch, labels_batch])git s

                predicted_class = sess.run(tf.nn.top_k(logits, k=1)[1][0])
                #label_true = sess.run(tf.nn.top_k(labels, k=1)[1][0])

                if predicted_class == labels:
                    correct += 1

                logger.info('Success rate: %.2f of %i examples', correct/i*100, i)

            logger.info("Eval ends...")


def main(argv=None):
    train()


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


if __name__ == "__main__":
    tf.app.run()
