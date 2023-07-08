from keras.layers import Input, LSTM, Dense, Embedding, GlobalMaxPool1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

# Cargar embeddings vectors pre entrenado
word2vec = {}
with open(os.path.join('./datasets/glove.6B.%sd.txt' % EMBEDDING_DIM), encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec


# Obtener Dataset
train = pd.read_csv('./datasets/toxic_comment_train.csv')
sentences = train['comment_text'].fillna("DUMMY_VALUE").values
possible_labels = train.columns[2:]
targets = train[possible_labels].values


# Convertir las frases a numeros
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Obtener el mapping de palabras a numeros
word2idx = tokenizer.word_index


# Padding para que todas las secuencias sean iguales (N x T)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Crear la embedding matrix
print("cantidad de palabras diferentes en nuestro dataset: ", len(word2idx))
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# Cargar modelo preentrenado de embedding
embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[
                            embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)


input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = LSTM(15, return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
output = Dense(len(possible_labels), activation="sigmoid")(x)

model = Model(input_, output)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.01), metrics=['accuracy'])

r = model.fit(data, targets, batch_size=BATCH_SIZE,
              epochs=EPOCHS, validation_split=VALIDATION_SPLIT)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
