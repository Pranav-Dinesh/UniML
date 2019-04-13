import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#Step 1: Gather your data
#Reshaping needs to be done since the convolution expects a single tensor, not 60000 tensors. We combine all of them into one.
training_images = training_images.reshape(60000, 28, 28, 1)
training_images/= 255.0

test_images = test_images.reshape(10000, 28, 28, 1)
test_images/= 255.0

#Step 2: Define your model
#Instead of the input layer at the top, we add a convolution

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D (32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
  #32 = no. of convolutions (arbitrary but in order of 32
  #Size of convolution = 3x3; Activation = RELU; Shape of the input data for the first layer
  tf.keras.layers.MaxPooling2D(2, 2),
  #To compress the image
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten()
  tf.keras.layers.Dense(128, activation='relu'
  tf.keras.layers.Dense(10, activation='softmax')
  ])
  
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
