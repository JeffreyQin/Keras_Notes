from tensorflow import keras
from keras.models import Sequential
from keras import Input
from keras.layers.experimental.preprocessing import TextVectorization
from keras.layers import Embedding, LSTM, Dense
from prepData import train_data, test_data

model = Sequential()
model.add(Input(shape=(1,), dtype='string'))

# vectorization layer

max_tokens = 1000
max_len = 100
vectorize_layer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_len
)

train_text = train_data.map(lambda text, label: text)
vectorize_layer.adapt(train_text)

model.add(vectorize_layer)

# embedding layer

model.add(Embedding(max_tokens + 1, 128))

## recurrent layer

model.add(LSTM(64))

model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# train model

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, epochs=10)


model.save_weights('rnn.h5')