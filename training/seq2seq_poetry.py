from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import os
import numpy as np
import matplotlib.pyplot as plt


# CONSTANTS

MAX_VOCAB_SIZE = 3000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 200
LATENT_DIM = 15
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
EPOCHS = 100

# load in the data
input_texts = []
target_texts = []
for line in open('./machine_learning_examples/hmm_class/robert_frost.txt'):
    line = line.rstrip()
    if not line:
        continue

    input_line = '<sos> ' + line
    target_line = line + ' <eos>'

    input_texts.append(input_line)
    target_texts.append(target_line)


all_lines = input_texts + target_texts


# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer.fit_on_texts(all_lines)
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# find max seq length
max_sequence_length_from_data = len(max(input_sequences, key=len))
print("max sequence: ", max_sequence_length_from_data)

# get word -> integer mapping
word2idx = tokenizer.word_index

# pad sequences so that we get a N x T matrix
max_sequence_length = min(max_sequence_length_from_data, MAX_SEQUENCE_LENGTH)
input_sequences = pad_sequences(input_sequences,
                                maxlen=max_sequence_length,
                                padding='post')
target_sequences = pad_sequences(target_sequences,
                                 maxlen=max_sequence_length,
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
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        vec = word2vec.get(word)
        if vec is not None:
            embedding_matrix[i] = vec

# one-hot the targets (can't use sparse cross-entropy)
one_hot_targets = np.zeros(
    (len(input_sequences), max_sequence_length, num_words))
for i, target_sequence in enumerate(target_sequences):
    for t, word in enumerate(target_sequence):
        if word > 0:
            one_hot_targets[i, t, word] = 1

embedding_layer = Embedding(
    num_words, EMBEDDING_DIM, weights=[embedding_matrix])


# build Model

input_ = Input(shape=(max_sequence_length,))
initial_h = Input(shape=(LATENT_DIM,))
initial_c = Input(shape=(LATENT_DIM,))

x = embedding_layer(input_)
lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
x, _, _ = lstm(x, initial_state=[initial_h, initial_c])
dense = Dense(num_words, activation='softmax')
output = dense(x)

model = Model([input_, initial_h, initial_c], output)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
z = np.zeros((len(input_sequences), LATENT_DIM))
results = model.fit(x=[input_sequences, z, z], y=one_hot_targets,
                    batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)

# Plot Losses
plt.plot(results.history['loss'], label='loss')
plt.plot(results.history['val_loss'], color='red', label='val_loss')
plt.legend()
plt.show()

# sampling model
input2 = Input(shape=(1,))  # one word at a time
x = embedding_layer(input2)
x, h, c = lstm(x, initial_state=[initial_h, initial_c])
output2 = dense(x)
sampling_model = Model([input2, initial_h, initial_c], [output2, h, c])

idx2word = {value: key for key, value in word2idx.items()}


def sample_line():
    # initial inputs
    np_input = np.array([[word2idx['<sos>']]])
    h = np.zeros((1, LATENT_DIM))
    c = np.zeros((1, LATENT_DIM))
    # so we know when to quit
    eos = word2idx['<eos>']

    # store the output here
    output_sentence = []

    for _ in range(max_sequence_length):
        o, h, c = sampling_model.predict([np_input, h, c], verbose="0")

        # print("o.shape:", o.shape, o[0,0,:10])
        # idx = np.argmax(o[0,0])
        probs = o[0, 0]
        # 0 is not a valid idx for word (word dict starts at 1)
        if np.argmax(probs) == 0:
            print("wtf")
        probs[0] = 0
        # Re-Normalize probs because we eliminated the values for 0
        probs /= probs.sum()
        # p = probabilidades asociadas a las palabras
        idx = np.random.choice(len(probs), p=probs)
        if idx == eos:
            break

        # accuulate output
        output_sentence.append(idx2word.get(idx, '<WTF %s>' % idx))

        # make the next input into model
        np_input[0, 0] = idx

    return ' '.join(output_sentence)


# generate a 4 line poem
while True:
    for _ in range(4):
        print(sample_line())

    ans = input("---generate another? [Y/n]---")
    if ans and ans[0].lower().startswith('n'):
        break
