#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
from absl import flags, app;
import tensorflow as tf;

FLAGS = flags.FLAGS;
flags.DEFINE_string('checkpoint', 'checkpoints/ckpt', help = 'path to checkpoint directory');

def minimize(_, loss):
  return loss;

def main(unused_argv):
  trainer = tf.keras.models.load_model(FLAGS.checkpoint, custom_objects = {'tf': tf, 'minimize': minimize}, compile = True);
  gen2 = trainer.get_layer('gen2');
  gen1 = trainer.get_layer('gen1');
  gen0 = trainer.get_layer('gen0');
  if not exists('models'): mkdir('models');
  gen2.save(join('models', 'gen2.h5'));
  gen1.save(join('models', 'gen1.h5'));
  gen0.save(join('models', 'gen0.h5'));

if __name__ == "__main__":
  app.run(main);
