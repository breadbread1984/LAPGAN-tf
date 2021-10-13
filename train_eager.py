#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
from absl import app, flags;
import numpy as np;
import cv2;
import tensorflow as tf;
from create_dataset import load_datasets;
from models import *;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_integer('batch_size', default = 256, help = 'batch size');
  flags.DEFINE_string('checkpoint', default = 'checkpoints', help = 'directory for checkpoint');
  flags.DEFINE_float('lr', default = 3e-4, help = 'learning rate');
  flags.DEFINE_bool('save_model', default = False, help = 'whethet to save model');
  flags.DEFINE_integer('disc_train_steps', default = 5, help = 'how many discriminator training steps for each generator training step');
  flags.DEFINE_integer('checkpoint_steps', default = 1000, help = 'how many iterations for each checkpoint');
  flags.DEFINE_integer('eval_steps', default = 100, help = 'how many iterations for each evaluation');

def main(unused_argv):
  # 1) create dataset
  trainset, testset = load_datasets();
  trainset = iter(trainset.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat(-1));
  testset = iter(testset.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat(-1));
  # 2) create model
  gen2 = GeneratorTwo();
  gen1 = GeneratorOne();
  gen0 = GeneratorZero();
  disc2 = DiscriminatorTwo();
  disc1 = DiscriminatorOne();
  disc0 = DiscriminatorZero();
  # 3) optimizer
  optimizer = tf.keras.optimizers.Adam(FLAGS.lr);
  # 4) restore from existing checkpoint
  if not exists(FLAGS.checkpoint): mkdir(FLAGS.checkpoint);
  checkpoint = tf.train.Checkpoint(gen2 = gen2, gen1 = gen1, gen0 = gen0, disc2 = disc2, disc1 = disc1, disc0 = disc0, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint(join(FLAGS.checkpoint, 'ckpt')));
  if FLAGS.save_model:
    gen2.save(join('models', 'gen2.h5'));
    gen1.save(join('models', 'gen1.h5'));
    gen0.save(join('models', 'gen0.h5'));
    disc2.save(join('models', 'disc2.h5'));
    disc1.save(join('models', 'disc1.h5'));
    disc0.save(join('models', 'disc0.h5'));
    exit();
  # 5) log
  log = tf.summary.create_file_writer('checkpoints');
  gen0_loss = tf.keras.metrics.Mean(name = 'gen0_loss', dtype = tf.float32);
  gen1_loss = tf.keras.metrics.Mean(name = 'gen1_loss', dtype = tf.float32);
  gen2_loss = tf.keras.metrics.Mean(name = 'gen2_loss', dtype = tf.float32);
  disc0_loss = tf.keras.metrics.Mean(name = 'disc0_loss', dtype = tf.float32);
  disc1_loss = tf.keras.metrics.Mean(name = 'disc1_loss', dtype = tf.float32);
  disc2_loss = tf.keras.metrics.Mean(name = 'disc2_loss', dtype = tf.float32);
  while True:
    (real_gaussian0, real_gaussian1, real_gaussian2, real_laplacian0, real_laplacian1), _ = next(trainset);
    noise2 = tf.random.normal(shape = (FLAGS.batch_size, 100), stddev = 0.1);
    noise1 = tf.random.normal(shape = (FLAGS.batch_size, 16,16,1), stddev = 0.1);
    noise0 = tf.random.normal(shape = (FLAGS.batch_size, 32,32,1), stddev = 0.1);
    with tf.GradientTape(persistent = False) as tape:
      fake_gaussian2 = gen2(noise2);
      fake_laplacian1 = gen1([noise1, real_gaussian1]);
      fake_laplacian0 = gen0([noise0, real_gaussian0]);
      fake_disc2 = disc2(fake_gaussian2);
      real_disc2 = disc2(real_gaussian2);
      fake_disc1 = disc1([fake_laplacian1, real_gaussian1]);
      real_disc1 = disc1([real_laplacian1, real_gaussian1]);
      fake_disc0 = disc0([fake_laplacian0, real_gaussian0]);
      real_disc0 = disc0([real_laplacian0, real_gaussian0]);
      d2_loss = 0.5 * (tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.zeros_like(fake_disc2), fake_disc2) + \
                       tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(real_disc2), real_disc2));
      d1_loss = 0.5 * (tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.zeros_like(fake_disc1), fake_disc1) + \
                       tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(real_disc1), real_disc1));
      d0_loss = 0.5 * (tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.zeros_like(fake_disc0), fake_disc0) + \
                       tf.kersa.losses.BinaryCrossentropy(from_logtis = False)(tf.ones_like(real_disc0), real_disc0));
      g2_loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(fake_disc2), fake_disc2);
      g1_loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(fake_disc1), fake_disc1);
      g0_loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(fake_disc0), fake_disc0);
    # train disc
    d2_grads = tape.gradient(d2_loss, disc2.trainable_variables);
    d1_grads = tape.gradient(d1_loss, disc1.trainable_variables);
    d0_grads = tape.gradient(d0_loss, disc0.trainable_variables);
    optimizer.apply_gradient(zip(d2_grads + d1_grads + d0_grads, disc2.trainable_variables + disc1.trainable_variables + disc0.trainable_variables));
    disc0_loss.update_state(d0_loss);
    disc1_loss.update_state(d1_loss);
    disc2_loss.update_state(d2_loss);
    # train gen
    if optimizer.iterations % FLAGS.disc_train_steps == 0:
      g2_grads = tape.gradient(g2_loss, gen2.trainable_variables);
      g1_grads = tape.gradient(g1_loss, gen1.trainable_variables);
      g0_grads = tape.gradeint(g0_loss, gen0.trainable_variables);
      optimizer.apply_gradient(zip(g2_grads + g1_grads + g0_grads, gen2.trainable_variables + gen1.trainable_variables + gen0.trainable_variables));
      gen0_loss.update_state(g0_loss);
      gen1_loss.update_state(g1_loss);
      gen2_loss.update_state(g2_loss);
    if tf.equal(optimizer.iterations % FLAGS.checkpoint_steps, 0):
      checkpoint.save(join(FLAGS.checkpoint, 'ckpt'));
    if tf.equal(optimizer.iterations % FLAGS.eval_steps, 0):
      lapgan = LAPGAN(gen2 = gen2, gen1 = gen1, gen0 = gen0);
      noise2 = np.random.normal(size = (1,100), scale = 0.1);
      noise1 = np.random.normal(size = (1, 16,16,1), scale = 0.1);
      noise0 = np.random.normal(size = (1, 32,32,1), scale = 0.1);
      sample = lapgan([noise2, noise1, noise0]);
      sample = tf.cast(sample, dtype = tf.uint8);
      with log.as_default():
        tf.summary.scalar('disc0_loss', disc0_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('disc1_loss', disc1_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('disc2_loss', disc2_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('gen0_loss', gen0_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('gen1_loss', gen1_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('gen2_loss', gen2_loss.result(), step = optimizer.iterations);
        tf.summary.image('sample', sample, step = optimizer.iterations);
      print('#%d d0_loss: %f d1_loss: %f d2_loss: %f g0_loss: %f g1_loss: %f g2_loss: %f' % (optimizer.iterations, \
                                                                                             disc0_loss.result(), \
                                                                                             disc1_loss.result(), \
                                                                                             disc2_loss.result(), \
                                                                                             gen0_loss.result(), \
                                                                                             gen1_loss.result(), \
                                                                                             gen2_loss.result()));
      disc0_loss.reset_states();
      disc1_loss.reset_states();
      disc2_loss.reset_states();
      gen0_loss.reset_states();
      gen1_loss.reset_states();
      gen2_loss.reset_states();

if __name__ == "__main__":
  add_options();
  app.run(main);
