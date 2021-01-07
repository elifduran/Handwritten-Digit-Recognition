import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D 

# Downloading MNIST dataset from Keras API and seperate labels and images as train and test.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# I choose the number that we want
image_index = 6666

# The number is 7.
print(y_train[image_index])

# Visualize the number(7) that i have chosen in previous step with matplotlib library.
plt.imshow(x_train[image_index], cmap='Greys')

# Learning the shape of dataset to change it cnn.
print(x_train.shape)

# I need 4 dimentional arrays but we have 3d dimentional arrays now.
# So i reshape x_train and x_tests 3 dim arrays to 4 dim arrays to work with Keras API.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# i want the values are float number to get decimal points after divisions.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes to the max RGB value with divided by 255.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Creating a Sequential Model and adding the layers.
model = Sequential()
# Add Convolution layer for 2D.
# 28 is the  number of output filters in the convolution.
# kernel size(3x3) specifies the height and width of 2D convolution windows.
# input shape is 28x28 images as we found it previous steps.
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
# Add Pooling layer for 2D.
# Doing Max pooling approach for 2D spatial data.
# pool size is 2x2 and it shows window size over which to take the maximum.
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flattening the 2D arrays to converts the pooled feature map to a single column that is passed to the fully connected layer.
model.add(Flatten())
# Adding fully connected layer to Sequential Model with Dense.
# 128 is the number of nodes in hidden layer and relu is the activation function in the hidden layer.
model.add(Dense(128, activation=tf.nn.relu))
# Dropout prevents overfitting. 0.2 is the rate at each step during training time.
model.add(Dropout(0.2))
# Adding output layer to Sequential Model with Dense.
# Activation function is chosen softmax function to get two outcomes.
# i have 10 numbers (0,1,2,..,9) so i need 10 neurons.
# 10 is the number of classes/neurons.
model.add(Dense(10, activation=tf.nn.softmax))
# Setting an optimizer with a given loss function that uses a metric to empty CNN.
# loss function (sparse_categorical_crossentropy) computes the crossentropy loss between the labels and predictions.
# The optimizer is (adam) the gradient descent algorithm.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Fitting model by using train data.
# 2 is the number of complete passes through the training dataset.
model.fit(x=x_train, y=y_train, epochs=2)
# Evaluating trained model with x_test and y_test.
print(model.evaluate(x_test, y_test))
