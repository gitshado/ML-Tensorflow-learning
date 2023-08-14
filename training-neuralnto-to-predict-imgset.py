import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import math

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.6): # Experiment with changing this value
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True


def call():
  callbacks = myCallback()
  dataset=tf.keras.datasets.fashion_mnist
  (trn_img,trn_lb),(test_img,test_lb)=dataset.load_data()

  index = 0

  # Set number of characters per row when printing
  np.set_printoptions(linewidth=320)

  # Print the label and image
  #print(f'LABEL: {trn_lb[index]}')
  #print(f'\nIMAGE PIXEL ARRAY:\n {trn_img[index]}')

  # Visualize the image
  ##iip= np.array(trn_img[index],dtype='float').reshape((28,28))
  ##plt.imshow(iip,cmap='gray')
  ##plt.show()
  trn_img = trn_img/255.0
  test_img = test_img/255.0

  # model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
  #                                     # Add a layer here,
  #                                     tf.keras.layers.Dense(256, activation=tf.nn.relu),
  #                                     # Add a layer here
  #                                   ])
  
  model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(trn_img, trn_lb, epochs=5, callbacks=[callbacks])

  # model.compile(optimizer = 'adam',
  #               loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

  # model.fit(trn_img, trn_lb, epochs=5)

  model.evaluate(test_img, test_lb)

  classifications = model.predict(test_img)

  # plt.imshow(np.array(classifications[0],dtype='float').reshape((28,28)))
  # plt.show()
  print(test_lb[0])
call()
# inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
# inputs = tf.convert_to_tensor(inputs)
# print(f'input to softmax function: {inputs.numpy()}')

# # Feed the inputs to a softmax activation function
# outputs = tf.keras.activations.softmax(inputs)
# print(f'output of softmax function: {outputs.numpy()}')
# print(tf.reduce_sum(math.exp(1.0)))
# # Get the sum of all values after the softmax
# sum = tf.reduce_sum(outputs)
# print(f'sum of outputs: {sum}')

# # Get the index with highest value
# prediction = np.argmax(outputs)
# print(f'class with highest probability: {prediction}')
