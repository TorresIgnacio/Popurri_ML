import os
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

from transformer_classes import Transformer, CustomSchedule, masked_loss, masked_accuracy, Decoder, PositionalEmbedding

current_path = os.path.dirname(__file__)
DATASET_PATH = os.path.join(
    current_path, '../../datasets/')
MODEL_PATH = os.path.join(
    current_path, '../../models/translator/ted_hrlr_translate_pt_es_converter')

tokenizers = tf.saved_model.load(MODEL_PATH)
print('\n\n')
print([item for item in dir(tokenizers.es) if not item.startswith('_')])

examples, metadata = tfds.load('ted_hrlr_translate/es_to_pt',
                               data_dir=DATASET_PATH,
                               with_info=True,
                               as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

for es_examples, pt_examples in train_examples.cache().batch(3).take(1):
    print('> Examples in Portuguese:')
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'))
        print()

    print('> Examples in Spanish:')
    for es in es_examples.numpy():
        print(es.decode('utf-8'))

encoded = tokenizers.es.tokenize(es_examples)
print('\n> This is a padded-batch of token IDs:')
for row in encoded.to_list():
    print(row)

round_trip = tokenizers.es.detokenize(encoded)
print('\n> This is human-readable text:')
for line in round_trip.numpy():
    print(line.decode('utf-8'))

print('\n> This is the text split into tokens:')
tokens = tokenizers.es.lookup(encoded)
print(tokens)

lengths = []

for es_examples, pt_examples in train_examples.batch(1024):
    pt_tokens = tokenizers.pt.tokenize(pt_examples)
    lengths.append(pt_tokens.row_lengths())

    es_tokens = tokenizers.es.tokenize(es_examples)
    lengths.append(es_tokens.row_lengths())
    print('▓', end='', flush=True)

all_lengths = np.concatenate(lengths)

plt.hist(all_lengths, np.linspace(0, 500, 101))
max_length = max(all_lengths)
plt.plot([max_length, max_length], plt.ylim())
plt.title(f'Maximum tokens per example: {max_length}')
# plt.show()

MAX_TOKENS = 128


def prepare_batch(es, pt):
    es = tokenizers.es.tokenize(es)      # Output is ragged.
    es = es[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    es = es.to_tensor()  # Convert to 0-padded dense Tensor

    pt = tokenizers.pt.tokenize(pt)
    pt = pt[:, :(MAX_TOKENS+1)]
    pt_inputs = pt[:, :-1].to_tensor()  # Drop the [END] tokens
    pt_labels = pt[:, 1:].to_tensor()   # Drop the [START] tokens

    return (es, pt_inputs), pt_labels


BUFFER_SIZE = 20000
BATCH_SIZE = 64


def make_batches(ds: tf.data.Dataset):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

for (es, pt), pt_labels in train_batches.take(1):
    break


num_layers = 2
d_model = 128
dff = 256
num_heads = 4
dropout_rate = 0.1


transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.es.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    dropout_rate=dropout_rate)

output = transformer((es, pt))

print(es.shape)
print(pt.shape)
print(output.shape)  # (batch_size, target_len, target_vocab_size)

print(transformer.summary())

attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
plt.ylabel('Learning Rate')
plt.xlabel('Train Step')

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)

transformer.fit(train_batches, epochs=20, validation_data=val_batches)

# Instantiate the decoder.
# sample_decoder = Decoder(num_layers=4,
#                          d_model=512,
#                          num_heads=8,
#                          dff=2048,
#                          vocab_size=8000)
#
# embed_pt = PositionalEmbedding(
#     vocab_size=tokenizers.pt.get_vocab_size(), d_model=512)
# embed_es = PositionalEmbedding(
#     vocab_size=tokenizers.es.get_vocab_size(), d_model=512)
#
# pt_emb = embed_pt(pt)
# es_emb = embed_es(es)
#
# output = sample_decoder(
#     x=pt,
#     context=es_emb)
#
# # Print the shapes.
# print(pt.shape)
# print(es_emb.shape)
# print(output.shape)
