import json
import matplotlib.pyplot as plt
from keras.backend import flatten
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model
from keras.utils import pad_sequences, to_categorical
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.layers import Concatenate, Dense, Dot, Input, Layer, RepeatVector

MODEL_PATH = './models/translator/translator_with_attention.keras'
LOSS_PATH = './evaluation/translator_with_attention_loss_40_to_140.png'
TOKENIZER_INPUTS_PATH = './training/tokenizer_inputs.json'
TOKENIZER_OUTPUTS_PATH = './training/tokenizer_outputs.json'
CHECKPOINT_PATH = './models/translator/translator_with_attention_checkpoint.keras'


NUM_TRAINING_SAMPLES = 20000
TRAINING_SAMPLES_START = 0
NUM_TEST_SAMPLES = 5000
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 30
BATCH_SIZE = 256
EPOCHS = 100


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
max_len_target = model.get_layer('decoder_input').input.shape[1]


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

    if t >= NUM_TRAINING_SAMPLES + TRAINING_SAMPLES_START:
        break

    if t >= TRAINING_SAMPLES_START:
        line = line.rstrip()

        if '\t' not in line:
            continue

        input_text, translation, _ = line.split('\t')

        target_line = translation + ' <eos>'
        target_line_input = '<sos> ' + translation

        input_texts.append(input_text)
        target_texts.append(target_line)
        target_texts_inputs.append(target_line_input)

    t += 1


# convert the sentences (strings) into integers
print("t = ", t)
print("target_texts len = ", len(target_texts))

# load tokenizer jsons

with open(TOKENIZER_INPUTS_PATH, encoding='utf-8') as f:
    data = json.load(f)
    tokenizer_inputs = tokenizer_from_json(data)


with open(TOKENIZER_OUTPUTS_PATH, encoding='utf-8') as f:
    data = json.load(f)
    tokenizer_outputs = tokenizer_from_json(data)

input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(
    target_texts_inputs)

# get word -> integer mapping
word2idx_inputs = tokenizer_inputs.word_index
word2idx_outputs = tokenizer_outputs.word_index
idx2word_trans = {v: k for k, v in word2idx_outputs.items()}

# pad sequences so that we get a N x T matrix
max_len_input = model.get_layer('encoder_input').input.shape[1]

encoder_inputs = pad_sequences(input_sequences,
                               maxlen=max_len_input)

decoder_inputs = pad_sequences(
    target_sequences_inputs, maxlen=max_len_target, padding='post')

decoder_targets = pad_sequences(target_sequences,
                                maxlen=max_len_target,
                                padding='post',
                                truncating='post')

num_words_output = len(word2idx_outputs) + 1
decoder_targets_one_hot = to_categorical(
    decoder_targets, num_classes=num_words_output)

initial_s = model.get_layer('s0')
latent_dim_decoder = initial_s.input.shape[1]
initial_c = model.get_layer('c0')
z = np.zeros((len(encoder_inputs), latent_dim_decoder))

model_checkpoint = ModelCheckpoint(
    filepath=CHECKPOINT_PATH, save_weights_only=False, monitor='val_loss', save_best_only=True)

r = model.fit([encoder_inputs, decoder_inputs, z, z], decoder_targets_one_hot,
              batch_size=BATCH_SIZE, validation_split=0.2, epochs=EPOCHS, shuffle=True, use_multiprocessing=True, verbose=1, callbacks=[model_checkpoint])

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss', color='red')
plt.legend()

plt.savefig(LOSS_PATH)
model.save('./models/translator/translator_with_attention_toda_la_noche.keras')
plt.show()
