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
  kernel = tf.keras.layers.Lambda(lambda x: tf.constant(
    [[1./256., 4./256., 6./256., 4./256., 1./256.],
     [4./256., 16./256., 24./256., 16./256., 4./256.],
     [6./256., 24./256., 36./256., 24./256., 6./256.],
     [4./256., 16./256., 24./256., 16./256., 4./256.],
     [1./256., 4./256., 6./256., 4./256., 1./256.]]))(inputs);
  kernel = tf.keras.layers.Lambda(lambda x,c: tf.tile(tf.reshape(x, (5,5,1,1)), (1,1,c,1)), arguments = {'c': channels})(kernel); # kernel.shape = (5,5,channels,1)
  gaussian = tf.keras.layers.Lambda(lambda x: tf.nn.depthwise_conv2d(x[0], x[1] * 4, strides = [1,1,1,1], padding = 'SAME'))([results, kernel]);
  return tf.keras.Model(inputs = inputs, outputs = gaussian);

def Trainer():
  real_gaussian2 = tf.keras.Input((8,8,3)); # real_gaussian2.shape = (batch, 8, 8, 3)
  real_gaussian1 = tf.keras.Input((16,16,3)); # real_gaussian1.shape = (batch, 16, 16, 3)
  real_gaussian0 = tf.keras.Input((32,32,3)); # real_gaussian0.shape = (batch, 32, 32, 3)
  real_laplacian1 = tf.keras.Input((16,16,3)); # real_laplacian1.shape = (batch, 16, 16, 3)
  real_laplacian0 = tf.keras.Input((32,32,3)); # real_laplacian0.shape = (batch, 32, 32, 3)
  noise2 = tf.keras.layers.Lambda(lambda x: tf.random.normal(shape = (tf.shape(x)[0], 100), stddev = 0.1))(real_gaussian2);
  noise1 = tf.keras.layers.Lambda(lambda x: tf.random.normal(shape = (tf.shape(x)[0], 16,16,1), stddev = 0.1))(real_gaussian1);
  noise0 = tf.keras.layers.Lambda(lambda x: tf.random.normal(shape = (tf.shape(x)[0], 32,32,1), stddev = 0.1))(real_gaussian0);
  fake_gaussian2 = GeneratorTwo()(noise2); # fake_gaussian2.shape = (batch, 8, 8, 3)
  fake_laplacian1 = GeneratorOne()([noise1, real_gaussian1]); # fake_laplacian1.shape = (batch, 16, 16, 3)
  fake_laplacian0 = GeneratorZero()([noise0, real_gaussian0]); # fake_laplacian0.shape = (batch, 32, 32, 3)
  gaussian2 = tf.keras.layers.Concatenate(axis = 0)([real_gaussian2, fake_gaussian2]); # gaussian2.shape = (batch * 2, 8, 8, 3)
  gaussian1 = tf.keras.layers.Concatenate(axis = 0)([real_gaussian1, real_gaussian1]); # gaussian1.shape = (batch * 2, 8, 8, 3)
  laplacian1 = tf.keras.layers.Concatenate(axis = 0)([real_laplacian1, fake_laplacian1]); # laplacian1.shape = (batch * 2, 16, 16, 3)
  gaussian0 = tf.keras.layers.Concatenate(axis = 0)([real_gaussian0, real_gaussian0]); # gaussian0.shape = (batch * 2, 16, 16, 3)
  laplacian0 = tf.keras.layers.Concatenate(axis = 0)([real_laplacian0, fake_laplacian0]); # laplacian0.shape = (batch * 2, 32, 32, 3)
  disc2 = DiscriminatorTwo()([gaussian2]); # disc2.shape = (batch * 2, 1)
  disc1 = DiscriminatorOne()([laplacian1, gaussian1]); # disc1.shape = (batch * 2, 1)
  disc0 = DiscriminatorZero()([laplacian0, gaussian0]); # disc0.shape = (batch * 2, 1)
  loss2 = tf.keras.layers.Lambda(lambda x: tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(x[0][:tf.shape(x[1])[0],...]), x[0][:tf.shape(x[1])[0],...]) + \
                                           tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.zeros_like(x[0][tf.shape(x[1])[0]:,...]), x[0][tf.shape(x[1])[0]:,...]),
                                 name = 'loss2')([disc2, real_gaussian2]);
  loss1 = tf.keras.layers.Lambda(lambda x: tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(x[0][:tf.shape(x[1])[0],...]), x[0][:tf.shape(x[1])[0],...]) + \
                                           tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.zeros_like(x[0][tf.shape(x[1])[0]:,...]), x[0][tf.shape(x[1])[0]:,...]),
                                 name = 'loss1')([disc1, real_gaussian1]);
  loss0 = tf.keras.layers.Lambda(lambda x: tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.ones_like(x[0][:tf.shape(x[1])[0],...]), x[0][:tf.shape(x[1])[0],...]) + \
                                           tf.keras.losses.BinaryCrossentropy(from_logits = False)(tf.zeros_like(x[0][tf.shape(x[1])[0]:,...]), x[0][tf.shape(x[1])[0]:,...]),
                                 name = 'loss0')([disc0, real_gaussian0]);
  return tf.keras.Model(inputs = (real_gaussian0, real_gaussian1, real_gaussian2, real_laplacian0, real_laplacian1), outputs = (loss0, loss1, loss2));

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
  trainer = Trainer();
  tf.keras.utils.plot_model(trainer, to_file = 'trainer.png', expand_nested = True);
