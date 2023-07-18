from tensorflow.python.keras import backend as K
from keras.layers import LSTM, Concatenate, Dense, Embedding, Input, Bidirectional, Lambda, RepeatVector, Dot
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
LATENT_DIM_DECODER = 256
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
EPOCHS = 20
NUM_SAMPLES = 25000

MODEL_PATH = './models/translator/complete_model2.keras'


def softmax_over_time(x):
    assert (K.ndim(x) > 2)
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e/s


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
######### ENCODER LAYERS #########
encoder_inputs_placeholder = Input(
    shape=(max_len_input,), name='encoder_input')
encoder_outputs = embeding_layer(encoder_inputs_placeholder)
encoder = Bidirectional(
    LSTM(units=LATENT_DIM, return_sequences=True, name='encoder_lstm'))
encoder_outputs = encoder(encoder_outputs)


######### ATTENTION LAYERS #########
# Attention layers need to be global because
# they will be repeated Ty times at the decoder
attn_repeat_layer = RepeatVector(max_len_input)
attn_concat_layer = Concatenate(axis=-1)
attn_dense1 = Dense(10, activation='tanh')
attn_dense2 = Dense(1, activation=softmax_over_time)
attn_dot = Dot(axes=1)  # to perform the weighted sum of alpha[t] * h[t]


def one_step_attention(h, st_1):
    st_1 = attn_repeat_layer(st_1)
    x = attn_concat_layer([h, st_1])
    x = attn_dense1(x)
    alphas = attn_dense2(x)
    context = attn_dot([alphas, h])
    return context


######### DECODER LAYERS #########
decoder_inputs_placeholder = Input(shape=(max_len_target,))
decoder_embedding = Embedding(num_words_spanish, EMBEDDING_DIM_SPANISH, weights=[
                              embedding_matrix_target], name='decoder_embedding')
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
decoder_lstm = LSTM(units=LATENT_DIM_DECODER,
                    return_state=True, name='decoder_lstm')
decoder_dense = LSTM(units=num_words_output,
                     activation='softmax', name='decoder_dense')

initial_s = Input(shape=(LATENT_DIM_DECODER,), name='s0')
initial_c = Input(shape=(LATENT_DIM_DECODER,), name='c0')
context_last_word_concat_layer = Concatenate(axis=2)


# Necesitamos obtener la salida en varios pasos Ty y en cada paso utilizar todos los Tx h's

s = initial_s
c = initial_c


outputs = []

for t in range(max_len_target):  # Ty times
    context = one_step_attention(encoder_outputs, s)

    # Necesitamos una nueva layer para cada Ty
    selector = Lambda(lambda x: x[:, t:t+1])
    xt = selector(decoder_inputs_x)

    # Combine
    decoder_lstm_input = context_last_word_concat_layer([context, xt])

    o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

    decoder_outputs = decoder_dense(o)
    outputs.append(decoder_outputs)


# Stackear todas las salidas para obtener tensor de T x N x D
# T = Timesteps
# N = Batches
# D = Vocabulario
# Transponer para que quede como N x T x D
def stack_and_transpose(x):
    # x is a list of length T, each element is a batch_size x output_vocab_size tensor
    x = K.stack(x)  # is now T x batch_size x output_vocab_size tensor
    # is now batch_size x T x output_vocab_size
    x = K.permute_dimensions(x, pattern=(1, 0, 2))
    return x


# make it a layer
stacker = Lambda(stack_and_transpose)
outputs = stacker(outputs)

# create the model
model = Model(
    inputs=[
        encoder_inputs_placeholder,
        decoder_inputs_placeholder,
        initial_s,
        initial_c,
    ],
    outputs=outputs
)


# Compile the model and train it
# With hot encoding
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

z = np.zeros((len(encoder_inputs), LATENT_DIM_DECODER))
model.summary()
r = model.fit([encoder_inputs, decoder_inputs, z, z], decoder_targets_one_hot,
              batch_size=BATCH_SIZE, validation_split=0.2, epochs=EPOCHS, shuffle=True, use_multiprocessing=True, verbose=2)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss', color='red')
plt.legend()
plt.show()

model.save(MODEL_PATH)
