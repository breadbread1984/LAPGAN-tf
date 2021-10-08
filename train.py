#!/usr/bin/python3

from os.path import join;
from absl import app, flags;
import tensorflow as tf;
from models import Trainer;
from create_datasets import load_datasets;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_integer('batch_size', 256, help = 'batch size');
  flags.DEFINE_string('checkpoint', 'checkpoints', help = 'directory for checkpoint');
  flags.DEFINE_integer('epochs', 25, help = 'epochs');
  flags.DEFINE_float('lr', 3e-4, help = 'learning rate');

def minimize(_, loss):
  return loss;

class SummaryCallback(tf.keras.callbacks.Callback):
  def __init__(self, trainer, eval_freq = 100):

      
def main(unused_argv):
  if exists(FLAGS.checkpoint):
    trainer = tf.keras.models.load_model(FLAGS.checkpoint, custom_objects = {'tf': tf, 'minimize': minimize}, compile = True);
    optimizer = trainer.optimizer;
  else:
    trainer = Trainer();
    optimizer = tf.keras.optimizers.Adam(FLAGS.lr);
    trainer.compile(optimizer = optimizer,
                    loss = {'loss0': minimize, 'loss1': minimize, 'loss2': minimize});
  trainset, testset = load_datasets();
  trainset = trainset.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = testset.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.checkpoint),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.checkpoint, 'ckpt'), save_freq = 1000),
    SummaryCallback(trainer);
  ];
  trainer.fit(trainset, epochs = FLAGS.epochs, validation_data = testset, callbacks = callbacks);

if __name__ == "__main__":
  add_options();
  app.run(main);
