#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_datasets as tfds;

def download():

  cifar10_builder = tfds.builder('cifar10');
  cifar10_builder.download_and_prepare();

def load_datasets():

  trainset = tfds.load(name = 'cifar10', split = 'train', download = False);
  testset = tfds.load(name = 'cifar10', split = 'test', download = False);
  return trainset, testset;

if __name__ == "__main__":

  download();
