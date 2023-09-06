import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Input

MODEL_PATH = './models/translator/complete_model2.keras'

NUM_TRAINING_SAMPLES = 25000
NUM_TEST_SAMPLES = 10000
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 100
model = load_model(MODEL_PATH, compile=True)

max_len_target = 5


def decode_sequence(input_seq, idx2word_trans):
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
            [target_seq] + states_value, verbose=0)

        idx = np.argmax(output_tokens[0, 0, :])
        print("idx = ", idx)
        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)


# load in the data
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

model = load_model(MODEL_PATH, compile=True)
encoder = model.get_layer('lstm')
decoder = model.get_layer('lstm_1')

encoder_inputs_placeholder = model.get_layer('input_1').input
_, h, c = encoder.output

encoder_model = Model(encoder_inputs_placeholder, [h, c])


decoder_state_input_h = Input(shape=(decoder.units,))
decoder_state_input_c = Input(shape=(decoder.units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_embedding = model.get_layer('embedding_1')
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

decoder_outputs, h, c = decoder(
    decoder_inputs_single_x, initial_state=decoder_states_inputs)

decoder_states = [h, c]

decoder_dense = model.get_layer('dense')
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs_single] +
                      decoder_states_inputs, [decoder_outputs] + decoder_states)

idx2word_eng = {v: k for k, v in word2idx_inputs.items()}
idx2word_trans = {v: k for k, v in word2idx_outputs.items()}

while True:
    i = np.random.choice(len(input_texts))
    input_seq = encoder_inputs[i: i+1]
    translation = decode_sequence(input_seq, idx2word_trans=idx2word_trans)
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
