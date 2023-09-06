from keras.layers import TextVectorization
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from translator_classes import Encoder, CrossAttention, Decoder, ShapeChecker, Translator, Export, load_data, tf_lower_and_split_punct, masked_loss, masked_acc

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

# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=BEST_MODEL_PATH,
#     save_weights_only=False,
#     monitor='val_masked_acc',
#     mode='max',
#     save_best_only=True)
#
model.load_weights(WEIGHTS_PATH)

# model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
#                                    "TextVectorization": TextVectorization,
#                                    "tf_lower_and_split_punct": tf_lower_and_split_punct})

history = model.fit(train_ds.repeat(), epochs=EPOCHS, steps_per_epoch=EPOCHS,
                    validation_data=val_ds, validation_steps=20)

plt.plot(history.history['masked_loss'], label='masked_loss')
plt.plot(history.history['val_masked_loss'], label='val_masked_loss')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()
plt.show()

inputs = [
    'Hace mucho frio aqui.',  # "It's really cold here."
    'Esta es mi vida.',  # "This is my life."
    'Su cuarto es un desastre.'  # "His room is a mess"
]


model.save_weights(WEIGHTS_PATH)
# tf.keras.models.save_model(model, MODEL_PATH, save_format='tf')
export = Export(model)
tf.saved_model.save(export, MODEL_PATH, signatures={
    'serving_default': export.translate})
