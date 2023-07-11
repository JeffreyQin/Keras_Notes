from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from prepData import test_images, test_labels
import numpy as np

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.load_weights('model.h5')

predictions = model.predict(test_images[:5])

print(np.argmax(predictions, axis=1))
print(test_labels[:5])