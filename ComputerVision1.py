import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#Normalizing the values
training_images  = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
#Flatten() changes the square to a 1-D array
#Softmax takes the largest value
#relu returns X if X is positive. Else, it returns 0.

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)
#Passes the two sets and returns the loss based on the trained model

classifications = model.predict(test_images)
print(classifications[0])
print("Versus")
print(test_labels[0])
#Remember that we had 10k test images. Classifications is an array corresponding to the output layer. classifications[0] corresponds to the first test item
#classifications[9999] would then correspond to the 9999th test item.
#print(test_labels[0]) returns 9 (the category), which seems to be in line with what we get if we execute print(classifications[0]).
#These numbers are a probability that the value being classified is the corresponding value.

#*******************************************
#Questions:
#1. Is classifications[10000] valid in the above case?
#2. Suppose I have an array of 25x25. Flattening it will result in ____________ array of size _______________.
#3. What happens if I increase the number of neurons in the dense layer to 1024?
#4. What happens if I change the number of neurons in the outer layer to some value other than 10? ERROR - no. of classes should match.
#5. Can I add an additional hidden layer? YES!
#6. Why did I normalize by 255? Since each pixel value ranges from 0 - 255.
