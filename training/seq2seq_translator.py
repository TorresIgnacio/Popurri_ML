from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    from tensorflow.python.keras import backend as K
    if len(K._get_available_gpus()) > 0:
        from keras.layers import CuDNNLSTM as LSTM
        from keras.layers import CuDNNGRU as GRU
except:
    pass


# CONSTANTS

MAX_VOCAB_SIZE = 20000
MAX_VOCAB_SIZE_SPANISH = 50000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
EMBEDDING_DIM_SPANISH = 300
LATENT_DIM = 256
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
EPOCHS = 20
NUM_SAMPLES = 25000

MODEL_PATH = './models/translator/complete_model2.keras'

# load in the data
input_texts = []
target_texts = []
target_texts_inputs = []
t = 0

for line in open('./datasets/spa.txt', encoding='utf8'):
    t += 1
    line = line.rstrip()
    if t > NUM_SAMPLES:
        break

    if '\t' not in line:
        continue

    input_text, translation, _ = line.split('\t')

    target_line = translation + ' <eos>'
    target_line_input = '<sos> ' + translation

    input_texts.append(input_text)
    target_texts.append(target_line)
    target_texts_inputs.append(target_line_input)


# convert the sentences (strings) into integers
tokenizer_inputs = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

tokenizer_outputs = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(
    target_texts_inputs)

# find max seq length
max_len_input = len(max(input_sequences, key=len))
print("max input sequence: ", max_len_input)


# get word -> integer mapping
word2idx_inputs = tokenizer_inputs.word_index
word2idx_outputs = tokenizer_outputs.word_index

# pad sequences so that we get a N x T matrix
max_len_input = min(max_len_input, MAX_SEQUENCE_LENGTH)

encoder_inputs = pad_sequences(input_sequences,
                               maxlen=max_len_input)

max_len_target = len(max(target_sequences, key=len))
print("max target sequence: ", max_len_target)

max_len_target = min(max_len_target, MAX_SEQUENCE_LENGTH)

decoder_inputs = pad_sequences(
    target_sequences_inputs, maxlen=max_len_target, padding='post')

decoder_targets = pad_sequences(target_sequences,
                                maxlen=max_len_target,
                                padding='post')

# load in pre-trained english word vectors
word2vec = {}
with open(os.path.join('./datasets/glove.6B.%sd.txt' % EMBEDDING_DIM), encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec

print("number of words = ", len(word2vec))

# prepare embedding matrix
num_words = min(MAX_VOCAB_SIZE, len(word2vec) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
    if i < MAX_VOCAB_SIZE:
        vec = word2vec.get(word)
        if vec is not None:
            embedding_matrix[i] = vec

embeding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[
                           embedding_matrix], input_length=max_len_input)


# load in pre-trained spanish word vectors
word2vec_spanish = {}
with open(os.path.join('./datasets/wiki.es.vec'), encoding='utf8') as f:
    f.readline()
    for i, line in enumerate(f):
        if i >= MAX_VOCAB_SIZE_SPANISH:
            break
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec_spanish[word] = vec

# prepare embedding matrix
num_words_spanish = min(MAX_VOCAB_SIZE_SPANISH, len(word2vec_spanish) + 1)
embedding_matrix_target = np.zeros((num_words_spanish, EMBEDDING_DIM_SPANISH))
for word, i in word2idx_outputs.items():
    if i < MAX_VOCAB_SIZE_SPANISH:
        vec = word2vec_spanish.get(word)
        if vec is not None:
            embedding_matrix_target[i] = vec

print("number of words = ", len(word2vec_spanish))
print("number of samples = ", NUM_SAMPLES)

# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
# one hot encoding

num_words_output = len(word2idx_outputs) + 1
decoder_targets_one_hot = to_categorical(decoder_targets)

# decoder_targets_one_hot = np.zeros(
#     (
#         len(input_texts),
#         max_len_target,
#         num_words_output
#     ),
#     dtype='float32'
# )

# assign the values
# for i, d in enumerate(decoder_targets):
#     for t, word in enumerate(d):
#         if word != 0:
#             decoder_targets_one_hot[i, t, word] = 1


# Build Model
# Encoder
encoder_inputs_placeholder = Input(shape=(max_len_input,))
encoder_outputs = embeding_layer(encoder_inputs_placeholder)
encoder = LSTM(units=LATENT_DIM, return_state=True)
encoder_outputs, h, c = encoder(encoder_outputs)

# Decoder
decoder_inputs_placeholder = Input(shape=(max_len_target,))
decoder_embedding = Embedding(num_words_spanish, EMBEDDING_DIM_SPANISH, weights=[
                              embedding_matrix_target])
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
decoder = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs_x, initial_state=[h, c])
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs_placeholder,
              decoder_inputs_placeholder], decoder_outputs)

# Compile the model and train it
# With hot encoding
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# Without hot encoding
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['sparse_categorical_accuracy']
# )

r = model.fit([encoder_inputs, decoder_inputs], decoder_targets_one_hot,
              batch_size=BATCH_SIZE, validation_split=0.2, epochs=EPOCHS, shuffle=True, use_multiprocessing=True, verbose=2)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss', color='red')
plt.legend()
plt.show()

model.save(MODEL_PATH)
