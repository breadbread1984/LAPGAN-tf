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
  noise = tf.keras.Input((32,32,1));
  gaussian = tf.keras.Input((32,32,3));
  results = tf.keras.layers.Concatenate(axis = -1)([gaussian, noise]);
  results = tf.keras.layers.Conv2D(128, kernel_size = (3,3), padding = 'same', activation = tf.keras.activations.relu)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.Conv2D(128, kernel_size = (3,3), padding = 'same', activation = tf.keras.activations.relu)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  laplacian = tf.keras.layers.Conv2D(3, kernel_size = (3,3), padding = 'same')(results);
  return tf.keras.Model(inputs = (noise, gaussian), outputs = laplacian);

def GeneratorOne():
  noise = tf.keras.Input((16,16,1));
  gaussian = tf.keras.Input((16,16,3));
  results = tf.keras.layers.Concatenate(axis = -1)([gaussian, noise]);
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same', activation = tf.keras.activations.relu)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.Conv2D(64, kernel_size = (3,3), padding = 'same', activation = tf.keras.activations.relu)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  laplacian = tf.keras.layers.Conv2D(3, kernel_size = (3,3), padding = 'same')(results);
  return tf.keras.Model(inputs = (noise, gaussian), outputs = laplacian);

def GeneratorTwo():
  noise = tf.keras.Input((100,));
  results = tf.keras.layers.Dense(1200, activation = tf.keras.activations.relu)(noise);
  results = tf.keras.layers.Dense(1200, activation = tf.keras.activations.sigmoid)(results);
  results = tf.keras.layers.Dense(8*8*3)(results);
  results = tf.keras.layers.Reshape((8,8,3))(results);
  return tf.keras.Model(inputs = noise, outputs = results);

def PyrUp(channels):
  inputs = tf.keras.Input((None, None, channels)); # inputs.shape = (batch, height, width, channel)
  # 1) dilate with zeros
  results = tf.keras.layers.Lambda(lambda x: tf.stack([x, tf.zeros_like(x)], axis = -1))(inputs); # results.shape = (batch, height, width, channel, 2)
  results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 1, 4, 2, 3)))(results); # results.shape = (batch, height, 2, width, channel)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[3], tf.shape(x)[4])))(results); # results.shape = (batch, height*2, width, channel)
  results = tf.keras.layers.Lambda(lambda x: tf.stack([x, tf.zeros_like(x)], axis = -1))(results); # results.shape = (batch, height * 2, width, channels, 2)
  results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 1, 2, 4, 3)))(results); # results.shape = (batch, height * 2, width, 2, channels)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], -1, tf.shape(x)[4])))(results); # results.shape = (batch, height * 2, width * 2, channels);
  results = tf.keras.layers.Lambda(lambda x: x[:,:-1,:-1,:])(results); # results.shape = (batch, height * 2 - 1, width * 2 - 1, channels)
  # 2) reflect padding
  results = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "REFLECT"))(results);
  # 3) gaussian filtering
  #gaussian = tf.keras.layers.Lambda(lambda x: )(results)
  return tf.keras.Model(inputs = inputs, outputs = results);

def Trainer():
  noise2 = tf.keras.Input((100,));
  noise1 = tf.keras.Input((16,16,1));
  noise0 = tf.keras.Input((32,32,1));
  sample0 = GeneratorTwo()(noise2);
#  sample1 = 

if __name__ == "__main__":

  import numpy as np;
  disc_0 = DiscriminatorZero();
  results = disc_0([np.random.normal(size = (4,32,32,3)), np.random.normal(size = (4,32,32,3))]);
  print(results.shape);
  disc_1 = DiscriminatorOne();
  results = disc_1([np.random.normal(size = (4,16,16,3)), np.random.normal(size = (4,16,16,3))]);
  print(results.shape);
  disc_2 = DiscriminatorTwo();
  results = disc_2([np.random.normal(size = (4,8,8,3))]);
  print(results.shape);
  gen_0 = GeneratorZero();
  results = gen_0([np.random.normal(size = (4,32,32,1)), np.random.normal(size = (4,32,32,3))]);
  print(results.shape);
  gen_1 = GeneratorOne();
  results = gen_1([np.random.normal(size = (4,16,16,1)), np.random.normal(size = (4,16,16,3))]);
  print(results.shape);
  gen_2 = GeneratorTwo();
  results = gen_2(np.random.normal(size = (4,100)));
  print(results.shape);
  pyrup = PyrUp(2);
  results = pyrup(tf.ones((1,2,2,2)));
  print(results[0,:,:,0]);
  print(results[0,:,:,1]);
