import json
from tensorflow.python.keras import backend as K
from keras.preprocessing.text import Tokenizer
import pandas as pd

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
NUM_SAMPLES = 1000000
TRAINING_SAMPLES_START = 0

TOKENIZER_INPUTS_PATH = './training/tokenizer_inputs.json'
TOKENIZER_OUTPUTS_PATH = './training/tokenizer_outputs.json'
PARAMETERS_PATH = './training/parameters.json'

dataset_parameters = {}

# load in the data
input_texts = []
target_texts = []
target_texts_inputs = []
t = 0

for line in open('./datasets/spa.txt', encoding='utf8'):

    if t >= NUM_SAMPLES + TRAINING_SAMPLES_START:
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
tokenizer_inputs = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

tokenizer_outputs = Tokenizer(num_words=MAX_VOCAB_SIZE_SPANISH, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(
    target_texts_inputs)

max_len_input = len(max(input_sequences, key=len))
max_len_target = len(max(target_sequences, key=len))
dataset_parameters['max_len_input'] = max_len_input
dataset_parameters['max_len_target'] = max_len_target

# with open(TOKENIZER_INPUTS_PATH, 'w', encoding='utf-8') as f:
#     tokenizer_json = tokenizer_inputs.to_json()
#     json.dump(tokenizer_json, f, ensure_ascii=False, indent=4)
#
# with open(TOKENIZER_OUTPUTS_PATH, 'w', encoding='utf-8') as f:
#     tokenizer_json = tokenizer_outputs.to_json()
#     json.dump(tokenizer_json, f, ensure_ascii=False, indent=4)
#
# with open(PARAMETERS_PATH, 'w', encoding='utf-8') as f:
#     json.dump(dataset_parameters, f, ensure_ascii=False, indent=4)
