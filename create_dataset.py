#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;
from models import PyrUp;

def download():

  cifar10_builder = tfds.builder('cifar10');
  cifar10_builder.download_and_prepare();

def parse_function(features):
  _, img, label = features['id'], features['image'].numpy(), features['label'];
  gaussian = list();
  laplacian = list();
  for i in range(3):
    if i == 2:
      gaussian.append(img);
    else:
      downsampled = np.array([cv2.pyrDown(img[...,c]) for c in range(3)]); # downsampled.shape = (h, w, channel)
      coarsed = PyrUp(3)(tf.expand_dims(downsampled, axis = 0))[0]; # coarsed.shape = (h, w, channel)
      residual = tf.constant(img - coarsed.numpy());
      gaussian.append(coarsed);
      laplacian.append(residual);
      img = downsampled;
  return (*gaussian, *laplacian), (0,0,0);

def load_datasets():

  trainset = tfds.load(name = 'cifar10', split = 'train', download = False);
  testset = tfds.load(name = 'cifar10', split = 'test', download = False);
  return trainset, testset;

if __name__ == "__main__":

  download();
