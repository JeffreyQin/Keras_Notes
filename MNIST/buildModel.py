from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from prepData import train_images, train_labels, test_images, test_labels

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    batch_size=32
)

model.evaluate(
    test_images,
    to_categorical(test_labels)
)

model.save_weights('model.h5')