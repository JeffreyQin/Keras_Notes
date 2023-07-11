from tensorflow import keras
from keras.models import Sequential

model = Sequential()

model.load_weights('rnn.h5')

prediction = model.predict([
    "i loved it!"
])