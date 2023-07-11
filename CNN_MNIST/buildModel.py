from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from prepData import train_images, train_labels, test_images, test_labels

num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28,28,1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax')
])

model.compile(
    'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=3,
    validation_data=(test_images, to_categorical(test_labels))
)

model.save_weights('cnn.h5')