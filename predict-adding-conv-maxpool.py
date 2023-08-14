import tensorflow as tf
import numpy as np
from tensorflow import keras

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

model = tf.keras.models.Sequential([
                                                         
  # Add convolutions and max pooling
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),

  # Add the same layers as before
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Print the model summary
model.summary()

# Use same settings
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print(f'\nMODEL TRAINING:')
model.fit(trn_img, trn_lb, epochs=5)

# Evaluate on the test set
print(f'\nMODEL EVALUATION:')
test_loss = model.evaluate(test_img, test_lb)