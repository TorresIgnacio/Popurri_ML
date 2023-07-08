from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
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
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
LATENT_DIM = 256
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
EPOCHS = 30
NUM_SAMPLES = 10000

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
# load in pre-trained word vectors

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

# # one hot encoding
# decoder_targets_one_hot = to_categorical(decoder_targets)

# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
num_words_output = len(word2idx_outputs) + 1
decoder_targets_one_hot = np.zeros(
    (
        len(input_texts),
        max_len_target,
        num_words_output
    ),
    dtype='float32'
)

# assign the values
for i, d in enumerate(decoder_targets):
    for t, word in enumerate(d):
        if word != 0:
            decoder_targets_one_hot[i, t, word] = 1


# Build Model
# Encoder
encoder_inputs_placeholder = Input(shape=(max_len_input,))
encoder_outputs = embeding_layer(encoder_inputs_placeholder)
encoder = LSTM(units=LATENT_DIM, return_state=True)
encoder_outputs, h, c = encoder(encoder_outputs)

# Decoder
decoder_inputs_placeholder = Input(shape=(max_len_target,))
decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
decoder = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs_x, initial_state=[h, c])
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs_placeholder,
              decoder_inputs_placeholder], decoder_outputs)


def custom_loss(y_true, y_pred):
    # both are shape N x T x K
    mask = K.cast(y_true > 0, dtype='float32')
    out = mask * y_true * K.log(y_pred)
    return -K.sum(out) / K.sum(mask)


def acc(y_true, y_pred):
    targ = K.argmax(y_true, axis=-1)
    pred = K.argmax(y_pred, axis=-1)
    correct = K.cast(K.equal(targ, pred), dtype='float32')

    mask = K.cast(K.greater(targ, 0), dtype='float32')
    n_correct = K.sum(mask * correct)
    n_total = K.sum(mask)
    return n_correct / n_total


# model.compile(optimizer='adam', loss=custom_loss, metrics=[acc])
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

r = model.fit([encoder_inputs, decoder_inputs], decoder_targets_one_hot,
              batch_size=BATCH_SIZE, validation_split=0.2, epochs=EPOCHS, verbose="2")

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss', color='red')
plt.legend()
plt.show()

model.save('s2s.h5')


# Predictions


encoder_model = Model(encoder_inputs_placeholder, [h, c])

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

decoder_outputs, h, c = decoder(
    decoder_inputs_single_x, initial_state=decoder_states_inputs)

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs_single] +
                      decoder_states_inputs, [decoder_outputs] + decoder_states)

idx2word_eng = {v: k for k, v in word2idx_inputs.items()}
idx2word_trans = {v: k for k, v in word2idx_outputs.items()}


def decode_sequence(input_seq):
    # Encode input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']

    # Translation
    output_sentence = []
    for _ in range(max_len_target):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        idx = np.argmax(output_tokens[0, 0, :])
        print("idx = ", idx)
        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)
            print("word = ", word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)


while True:
    i = np.random.choice(len(input_texts))
    input_seq = encoder_inputs[i: i+1]
    translation = decode_sequence(input_seq)
    print(f"-\nInput: {input_texts[i]}\nTranslation: {translation}")

    ans = input("Continue? [Y/n]")
    if ans and ans.lower().startswith('n'):
        break
