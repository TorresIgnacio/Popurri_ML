from keras.layers import TextVectorization
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from training.translator_classes import Encoder, CrossAttention, Decoder, ShapeChecker, Translator, Export, load_data, tf_lower_and_split_punct, masked_loss, masked_acc
import sys

# adding Folder_2 to the system path
sys.path.insert(0, './training/')

BATCH_SIZE = 256
MAX_VOCAB_SIZE = 30000
UNITS = 256
EPOCHS = 1

DATASET_PATH = pathlib.Path('./datasets/spa.txt')
BEST_MODEL_PATH = './models/translator/best_translator.keras'
MODEL_PATH = './models/translator/translator2'
WEIGHTS_PATH = './models/translator/translator_weights'


input_raw, target_raw = load_data(DATASET_PATH)

BUFFER_SIZE = len(input_raw)

np.random.seed(101)
is_train = np.random.uniform(size=(len(target_raw),)) < 0.8

train_raw = (tf.data.Dataset
             .from_tensor_slices((input_raw[is_train], target_raw[is_train]))
             .shuffle(BATCH_SIZE)
             .batch(BATCH_SIZE))

val_raw = (tf.data.Dataset
           .from_tensor_slices((input_raw[~is_train], target_raw[~is_train]))
           .shuffle(BATCH_SIZE)
           .batch(BATCH_SIZE))

# Process input vocab (English)
input_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=MAX_VOCAB_SIZE,
    ragged=True
)
input_text_processor.adapt(train_raw.map(lambda inputs, _: inputs))

# Process target vocab (Spanish)
target_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=MAX_VOCAB_SIZE,
    ragged=True
)
target_text_processor.adapt(train_raw.map(lambda _, targets: targets))


def process_text(input_text, target_text):
    input_tokens = input_text_processor(input_text).to_tensor()
    target_tokens = target_text_processor(target_text)
    targ_in = target_tokens[:, :-1].to_tensor()
    targ_out = target_tokens[:, 1:].to_tensor()
    return (input_tokens, targ_in), targ_out


train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

model = Translator(UNITS, input_text_processor, target_text_processor)

model.compile(optimizer='adam',
              loss=masked_loss,
              metrics=[masked_acc, masked_loss])


model.load_weights(WEIGHTS_PATH)


inputs = [
    'Hace mucho frio aqui.',  # "It's really cold here."
    'Esta es mi vida.',  # "This is my life."
    'Su cuarto es un desastre.'  # "His room is a mess"
]

inputs = tf.constant(inputs)
print(model.translate(inputs).numpy())
