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
import input, model
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ckpt_dir', 'data/checkpoints/',
                           """Directory where to restore a model""")
tf.app.flags.DEFINE_string('save_dir', 'data/train/flowers',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('log_dir', 'data/train/log',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('max_steps', 500,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Size of batches.""")
tf.app.flags.DEFINE_float('learning_rate', 0.005,
                            """Learning rate for the optimizer""")


def train():
    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()

        # Get the iterator from the TFRecord files.
        iterator = input.consume_tfrecord()
        images_batch, labels_batch = iterator.get_next()

        # Num_classes is None for fine tuning. You need to have the proper scope.
        # From the original model we only need the bottlenecks.
        with tf.contrib.slim.arg_scope(model.inception_v3_arg_scope()):
            bottleneck, end_points = model.inception_v3(images_batch, num_classes=None, is_training=False)

        # We pass the bottleneck generated in the step before to the new classifier.
        logits = model.fine_tuning(bottleneck, end_points)

        # We compute the loss between the predictions and the labels
        loss = model.loss(logits, labels_batch)

        # We use ADAM as a optimizer. You could use whichever you want, like Gradient Descent.
        # It's important to indicate that we only want to retrain the 'fine_tuning' variables.
        optimizer = tf.train.AdamOptimizer(0.005)
        train_op = optimizer.minimize(loss, global_step=global_step, var_list=tf.global_variables('fine_tuning'))

        # We create two savers. The first one for the InceptionV3 variables and the second one for the variables of
        # the new classifier.
        saver = tf.train.Saver(tf.global_variables('InceptionV3'))
        saver_ft = tf.train.Saver(tf.global_variables('fine_tuning'))
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Restore the checkpoints of the InceptionV3 model.
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))

            # This will let you see the images in tensorboard
            tf.summary.image(tensor=images_batch, name="Image")

            # Tensorborad options
            train_writer = tf.summary.FileWriter(FLAGS.log_dir, g)

            logger = init_logger()
            logger.info("Training starts...")

            # Training loop. Set the max number of steps.
            for i in range(0, FLAGS.max_steps):
                # We compute the image and label batch
                sess.run([images_batch, labels_batch])

                # Merge all summary variables for Tensorborad
                merge = tf.summary.merge_all()

                # We do the training and compute the loss and the summaries
                _, loss_val, summary = sess.run([train_op, loss, merge])

                if i % 10 is 0:
                    logger.info('Time: %s Loss: %f Step: %i', datetime.now(), loss_val, i)
                    # Write the summaries in the log file
                    train_writer.add_summary(summary, i)

                # We save the progress every 500 steps
                if i % 500 is 0 and i is not 0:
                    saver_ft.save(sess, FLAGS.save_dir, global_step=global_step)
                    logger.info("***** Saving model in: %s *****", FLAGS.save_dir)

            logger.info("Training ends...")
            saver_ft.save(sess, FLAGS.save_dir, global_step=global_step)
            logger.info("***** Saving model in: %s *****", FLAGS.save_dir)


def main(argv=None):
    train()


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


if __name__ == "__main__":
    tf.app.run()
