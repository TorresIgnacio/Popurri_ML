import matplotlib.pyplot as plt
from keras.backend import flatten
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from keras.models import load_model, Model
from keras.utils import pad_sequences, to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import Concatenate, Dense, Dot, Input, Layer, RepeatVector

MODEL_PATH = './models/translator/translator_with_attention2.keras'
LOSS_PATH = './evaluation/translator_with_attention_loss_40_to_60.png'

NUM_TRAINING_SAMPLES = 25000
NUM_TEST_SAMPLES = 10000
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100
BATCH_SIZE = 256
EPOCHS = 20


class Slice(Layer):
    def __init__(self, begin, size, **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.begin = begin
        self.size = size

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size,
        })
        return config

    def call(self, inputs):
        return tf.slice(inputs, self.begin, self.size)


def softmax_over_time(x):
    assert (K.ndim(x) > 2)
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e/s


model = load_model(MODEL_PATH, compile=True, custom_objects={
    "softmax_over_time": softmax_over_time, "Slice": Slice})
max_len_target = model.get_layer('encoder_input').input.shape[1]


def decode_sequence(input_seq, idx2word_trans, latent_dim_decoder):
    # Encode input as state vectors
    enc_out = encoder_model.predict(input_seq)

    # Empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']

    s = np.zeros((1, latent_dim_decoder))
    c = np.zeros((1, latent_dim_decoder))

    # Translation
    output_sentence = []
    for _ in range(max_len_target):
        o, s, c = decoder_model.predict([target_seq, enc_out, s, c])

        idx = np.argmax(flatten(o))
        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx

    return ' '.join(output_sentence)


# load in the data
print("load data")
input_texts = []
target_texts = []
target_texts_inputs = []
t = 0

for line in open('./datasets/spa.txt', encoding='utf8'):
    t += 1
    line = line.rstrip()
    if t > NUM_TRAINING_SAMPLES:
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
idx2word_trans = {v: k for k, v in word2idx_outputs.items()}

# pad sequences so that we get a N x T matrix
max_len_input = min(max_len_input, MAX_SEQUENCE_LENGTH)

encoder_inputs = pad_sequences(input_sequences,
                               maxlen=6)

max_len_target = len(max(target_sequences, key=len))
max_len_target = min(max_len_target, MAX_SEQUENCE_LENGTH)
decoder_inputs = pad_sequences(
    target_sequences_inputs, maxlen=max_len_target, padding='post')

decoder_targets = pad_sequences(target_sequences,
                                maxlen=max_len_target,
                                padding='post')
decoder_targets_one_hot = to_categorical(decoder_targets)

initial_s = model.get_layer('s0')
latent_dim_decoder = initial_s.input.shape[1]
initial_c = model.get_layer('c0')
z = np.zeros((len(encoder_inputs), latent_dim_decoder))

r = model.fit([encoder_inputs, decoder_inputs, z, z], decoder_targets_one_hot,
              batch_size=BATCH_SIZE, validation_split=0.2, epochs=EPOCHS, shuffle=True, use_multiprocessing=True, verbose=2)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss', color='red')
plt.legend()

plt.savefig(LOSS_PATH)
model.save(MODEL_PATH)
plt.show()
