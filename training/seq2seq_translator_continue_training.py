from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from tensorflow.python.keras import backend as K
    if len(K._get_available_gpus()) > 0:
        from keras.layers import CuDNNLSTM as LSTM
        from keras.layers import CuDNNGRU as GRU
except:
    pass


# CONSTANTS

MODEL_PATH = './models/translator/complete_model2.keras'
MAX_VOCAB_SIZE = 20000
MAX_VOCAB_SIZE_SPANISH = 40000
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
EPOCHS = 20
NUM_SAMPLES = 25000
FILE_LINES = 139705

model = load_model(MODEL_PATH, compile=True)


# load in the data
input_texts = []
target_texts = []
target_texts_inputs = []
start_line = np.random.choice(FILE_LINES - NUM_SAMPLES)
t = start_line

for line in open('./datasets/spa.txt', encoding='utf8'):
    t += 1
    line = line.rstrip()
    if t > (start_line + NUM_SAMPLES):
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
_, max_len_input = model.get_layer('input_1').get_config()['batch_input_shape']
print("max input sequence: ", max_len_input)


# get word -> integer mapping
word2idx_inputs = tokenizer_inputs.word_index
word2idx_outputs = tokenizer_outputs.word_index

# pad sequences so that we get a N x T matrix

encoder_inputs = pad_sequences(input_sequences,
                               maxlen=max_len_input)

_, max_len_target = model.get_layer('input_2').get_config()[
    'batch_input_shape']
print("max target sequence: ", max_len_target)


decoder_inputs = pad_sequences(
    target_sequences_inputs, maxlen=max_len_target, padding='post')

decoder_targets = pad_sequences(target_sequences,
                                maxlen=max_len_target,
                                padding='post')


print("number of samples = ", NUM_SAMPLES)


num_words_output = len(word2idx_outputs) + 1
decoder_targets_one_hot = to_categorical(decoder_targets)


# Compile the model and train it
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

r = model.fit([encoder_inputs, decoder_inputs], decoder_targets_one_hot,
              batch_size=BATCH_SIZE, validation_split=0.2, epochs=EPOCHS, shuffle=True, use_multiprocessing=True, verbose=2)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss', color='red')
plt.legend()
plt.show()

model.save(MODEL_PATH)
