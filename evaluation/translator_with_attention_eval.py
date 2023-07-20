from keras.backend import flatten
import numpy as np
from tensorflow.python.keras import backend as K
from keras.models import load_model, Model
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Concatenate, Dense, Dot, Input, RepeatVector

MODEL_PATH = './models/translator/translator_with_attention.keras'

NUM_TRAINING_SAMPLES = 25000
NUM_TEST_SAMPLES = 10000
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100


def softmax_over_time(x):
    assert (K.ndim(x) > 2)
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e/s


model = load_model(MODEL_PATH, compile=False, custom_objects={
                   "softmax_over_time": softmax_over_time})
max_len_target = model.get_layer('input_1').input.shape[1]


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

# Models

print(model.get_config())

# Get layers from model
encoder = model.get_layer('bidirectional')
encoder_inputs_placeholder = model.get_layer('encoder_input').input
embedding_layer = model.get_layer('embedding')
initial_s = model.get_layer('s0')
initial_c = model.get_layer('c0')
decoder = model.get_layer('decoder_lstm')
decoder_embedding = model.get_layer('decoder_embedding')
decoder_lstm = model.get_layer('decoder_lstm')
decoder_dense = model.get_layer('decoder_dense')

# Build encoder model
x = embedding_layer(encoder_inputs_placeholder)
encoder_outputs = encoder(x)
encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)


# Build attention layers
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


# Build decoder Model
encoder_outputs_as_input = Input(
    shape=(max_len_input, encoder.output.shape[2]))  # Encoder.output.shape[2] = LATENT_DIM * 2
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
context_last_word_concat_layer = Concatenate(
    axis=2, name='decoder_concatenate')

context = one_step_attention(encoder_outputs_as_input, initial_s)
decoder_lstm_input = context_last_word_concat_layer(
    [context, decoder_inputs_single_x])

o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[
    initial_s, initial_c])
decoder_outputs = decoder_dense(o)

decoder_model = Model(
    inputs=[
        decoder_inputs_single,
        encoder_outputs_as_input,
        initial_s,
        initial_c
    ],
    outputs=[decoder_outputs, s, c]
)

idx2word_eng = {v: k for k, v in word2idx_inputs.items()}
idx2word_trans = {v: k for k, v in word2idx_outputs.items()}

while True:
    i = np.random.choice(len(input_texts))
    input_seq = encoder_inputs[i: i+1]
    translation = decode_sequence(
        input_seq, idx2word_trans=idx2word_trans, latent_dim_decoder=initial_s.input.shape[1])
    print(f"-\nInput: {input_texts[i]}\nTranslation: {translation}")

    ans = input("Continue? [Y/n]")
    if ans and ans.lower().startswith('n'):
        break

while True:
    input_text = [input("Ingresar frase en ingles\n")]
    input_seq = tokenizer_inputs.texts_to_sequences(input_text)
    input_seq = pad_sequences(input_seq, maxlen=max_len_input)

    translation = decode_sequence(input_seq, idx2word_trans=idx2word_trans)
    print(f"-\nInput: {input_text}\nTranslation: {translation}")

    ans = input("Continue? [Y/n]")
    if ans and ans.lower().startswith('n'):
        break
