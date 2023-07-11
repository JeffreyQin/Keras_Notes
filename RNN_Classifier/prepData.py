from tensorflow import keras
from keras.preprocessing import text_dataset_from_directory
from tensorflow.strings import regex_replace

def prepareData(dir):
    data = text_dataset_from_directory(dir)
    return data.map(
        lambda text, label: (regex_replace(text, '<br />', ' '), label)
    )

train_data = prepareData('./aclImdb/train')
test_data = prepareData('./aclImdb/test')
