#!/usr/bin/python3

import tensorflow as tf;

def DiscriminatorZero():
  laplacian = tf.keras.Input((32,32,3));
  gaussian = tf.keras.Input((32,32,3));
  results = tf.keras.layers.Add()([laplacian, gaussian]);
  results = tf.keras.layers.Conv2D(128, kernel_size = (5,5))(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.Conv2D(128, kernel_size = (5,5), strides = (2,2))(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.Flatten()(results);
  results = tf.keras.layers.Dense(units = 1, activation = tf.keras.activations.sigmoid)(results);
  return tf.keras.Model(inputs = (laplacian, gaussian), outputs = results);

def DiscriminatorOne():
  laplacian = tf.keras.Input((16,16,3));
  gaussian = tf.keras.Input((16,16,3));
  results = tf.keras.layers.Add()([laplacian, gaussian]);
  results = tf.keras.layers.Conv2D(64, kernel_size = (5,5))(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.Conv2D(64, kernel_size = (5,5), strides = (2,2))(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.Flatten()(results);
  results = tf.keras.layers.Dense(units = 1, activation = tf.keras.activations.sigmoid)(results);
  return tf.keras.Model(inputs = (laplacian, gaussian), outputs = results);

def DiscriminatorTwo():
  gaussian = tf.keras.Input((8,8,3));
  results = tf.keras.layers.Flatten()(gaussian);
  results = tf.keras.layers.Dense(units = 600)(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Dense(units = 600)(results);
  results = tf.keras.layers.LeakyReLU()(results);
  results = tf.keras.layers.Dense(units = 1, activation = tf.keras.activations.sigmoid)(results);
  return tf.keras.Model(inputs = gaussian, outputs = results);

def GeneratorZero():
  
