#!/usr/bin/python3

from os.path import join;
import numpy as np;
import cv2;
import tensorflow as tf;
from models import LAPGAN;

def main():
  gen0 = tf.keras.models.load_model(join('models', 'gen0.h5'), custom_objects = {'tf': tf});
  gen1 = tf.keras.models.load_model(join('models', 'gen1.h5'), custom_objects = {'tf': tf});
  gen2 = tf.keras.models.load_model(join('models', 'gen2.h5'), custom_objects = {'tf': tf});
  lapgan = LAPGAN(gen2, gen1, gen0);
  while True:
    noise2 = np.random.normal(scale = 0.1, size = (1,100));
    noise1 = np.random.normal(scale = 0.1, size = (1, 16, 16, 1));
    noise0 = np.random.normal(scale = 0.1, size = (1, 32, 32, 1));
    sample = lapgan([noise2, noise1, noise0]);
    sample = tf.cast(sample, dtype = tf.uint8);
    cv2.imshow('sample', sample[0].numpy());
    cv2.waitKey();

if __name__ == "__main__":
  main();
