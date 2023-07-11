import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from prepData import test_images, test_labels

num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax')
])

model.load_weights('cnn.h5')

predictions = model.predict(test_images[:5])

print(np.argmax(predictions, axis=1))

print(test_labels[:5])