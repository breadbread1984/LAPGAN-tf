#!/usr/bin/python3

from os.path import join;
from absl import app, flags;
import tensorflow as tf;
from models import Trainer, LAPGAN;
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
    gen0 = trainer.get_layer('gen0');
    gen1 = trainer.get_layer('gen1');
    gen2 = trainer.get_layer('gen2');
    self.optimizer = trainer.optimizer;
    self.lapgan = LAPGAN(gen2, gen1, gen0);
    self.loss0 = tf.keras.metrics.Mean(name = 'loss0', dtype = tf.float32);
    self.loss1 = tf.keras.metrics.Mean(name = 'loss1', dtype = tf.float32);
    self.loss2 = tf.keras.metrics.Mean(name = 'loss2', dtype = tf.float32);
    self.log = tf.summary.create_file_writer(FLAGS.checkpoint);
  def on_batch_begin(self, batch, logs = None):
    pass;
  def on_batch_end(self, batch, logs = None):
    noise2 = np.random.normal(scale = 0.1, size = (1,100));
    noise1 = np.random.normal(scale = 0.1, size = (1, 16, 16, 1));
    noise0 = np.random.normal(scale = 0.1, size = (1, 32, 32, 1));
    sample = self.lapgan([noise2, noise1, noise0]); # sample.shape = (1, 32, 32, 3)
    sample = tf.cast(sample, dtype = tf.uint8);
    self.loss0.update_state(logs['loss0']);
    self.loss1.update_state(logs['loss1']);
    self.loss2.update_state(logs['loss2']);
    if batch % self.eval_freq == 0:
      with self.log.as_default():
        tf.summary.scalar('avg_loss0', self.loss0.result(), step = self.optimizer.iterations);
        tf.summary.scalar('avg_loss1', self.loss1.result(), step = self.optimizer.iterations);
        tf.summary.scalar('avg_loss2', self.loss2.result(), step = self.optimizer.iterations);
        tf.summary.image('sample', sample, step = self.optimizer.iterations);
      self.loss0.reset_states();
      self.loss1.reset_states();
      self.loss2.reset_states();
  def on_epoch_begin(self, epoch, logs = None):
    pass;
  def on_epoch_end(self, epoch, logs = None):
    pass;

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
