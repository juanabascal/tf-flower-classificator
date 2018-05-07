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
import logging
import input
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_dir', 'data/checkpoints/',
                           """Directory where to restore a model""")
tf.app.flags.DEFINE_string('eval_dir', '/home/uc3m4/Documentos/Trained/ft_flowers/lr_ed_0.4',
                           """Directory where to restore the fine tuning model""")
tf.app.flags.DEFINE_string('save_dir', 'data/train/flowers',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('log_dir', 'data/train/log',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('max_steps', 2000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Size of batches.""")


def train():

    with tf.Graph().as_default():
        # With is_training=False, we get the image without distortions
        iterator = input.consume_tfrecord(is_training=False, batch_size=FLAGS.batch_size)
        images_batch, labels_batch = iterator.get_next()

        # Num_classes is None for fine tuning. You need to have the proper scope.
        # From the original model we only need the bottlenecks.
        with tf.contrib.slim.arg_scope(model.inception_v3_arg_scope()):
            bottleneck, end_points = model.inception_v3(images_batch, num_classes=None, is_training=False,
                                                        dropout_keep_prob=1)

        # We pass the bottleneck generated in the step before to the new classifier.
        logits = model.fine_tuning(bottleneck, end_points)

        # Get the class with the highest score
        predictions = tf.nn.top_k(logits, k=1)

        # For eval you need to restore the fine tuning model
        saver = tf.train.Saver(tf.global_variables('InceptionV3'))
        saver_ft = tf.train.Saver(tf.global_variables('fine_tuning'))

        init = tf.global_variables_initializer()

        logger = init_logger()
        logger.info("Eval starts...")

        with tf.Session() as sess:
            sess.run(init)

            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            saver_ft.restore(sess, tf.train.latest_checkpoint(FLAGS.eval_dir))

            success = 0
            total = 0
            exec_next_step = True
            while exec_next_step is True:
                try:
                    total += 1
                    predicted, image, label = sess.run([predictions, images_batch, labels_batch])

                    if predicted.indices[0][0] == label[0]:
                        success += 1
                    logger.info('Success rate: %.2f of %i examples', success / total * 100, total)
                except tf.errors.OutOfRangeError:
                    logger.info("Eval ends...")
                    exec_next_step = False


def main(argv=None):
    train()


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


if __name__ == "__main__":
    tf.app.run()
