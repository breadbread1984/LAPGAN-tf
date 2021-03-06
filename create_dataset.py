#!/usr/bin/python3

from absl import app, flags;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from models import PyrDown, PyrUp;

FLAGS = flags.FLAGS;

def add_options():
  flags.DEFINE_bool('test', False, help = 'test dataset input pipeline');

def download():

  cifar10_builder = tfds.builder('cifar10');
  cifar10_builder.download_and_prepare();

def parse_sample(features):
  _, img, label = features['id'], features['image'], features['label'];
  img = tf.cast(img, dtype = tf.float32);
  gaussian = list();
  laplacian = list();
  for i in range(3):
    if i == 2:
      gaussian.append(img);
    else:
      downsampled = PyrDown(3)(tf.expand_dims(img, axis = 0))[0]; # downsampled.shape = (h, w, channel)
      coarsed = PyrUp(3)(tf.expand_dims(downsampled, axis = 0))[0]; # coarsed.shape = (h, w, channel)
      residual = img - coarsed;
      gaussian.append(coarsed);
      laplacian.append(residual);
      img = downsampled;
  return (*gaussian, *laplacian), (0,0,0,0,0,0);

def load_datasets():

  trainset = tfds.load(name = 'cifar10', split = 'train', download = False);
  testset = tfds.load(name = 'cifar10', split = 'test', download = False);
  trainset = trainset.map(parse_sample);
  testset = testset.map(parse_sample);
  return trainset, testset;

def main(unused_argv):
  if FLAGS.test:
    import numpy as np;
    import cv2;
    trainset, testset = load_datasets();
    cv2.namedWindow('gaussian0', cv2.WINDOW_NORMAL);
    cv2.namedWindow('gaussian1', cv2.WINDOW_NORMAL);
    cv2.namedWindow('gaussian2', cv2.WINDOW_NORMAL);
    cv2.namedWindow('laplacian0', cv2.WINDOW_NORMAL);
    cv2.namedWindow('laplacian1', cv2.WINDOW_NORMAL);
    cv2.namedWindow('reconstruct', cv2.WINDOW_NORMAL);
    for sample, label in trainset:
      real_gaussian0, real_gaussian1, real_gaussian2, real_laplacian0, real_laplacian1 = sample;
      cv2.imshow('gaussian0', real_gaussian0.numpy().astype(np.uint8));
      cv2.imshow('gaussian1', real_gaussian1.numpy().astype(np.uint8));
      cv2.imshow('gaussian2', real_gaussian2.numpy().astype(np.uint8));
      cv2.imshow('laplacian0', real_laplacian0.numpy());
      cv2.imshow('laplacian1', real_laplacian1.numpy());
      cv2.imshow('reconstruct', (real_gaussian0.numpy() + real_laplacian0.numpy()).astype(np.uint8))
      cv2.waitKey();
  else:
    download();

if __name__ == "__main__":

  add_options();
  app.run(main);
