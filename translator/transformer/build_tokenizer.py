import tensorflow as tf
import tensorflow_text as text
import tensorflow_datasets as tfds
import os
import re
import pathlib

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

current_path = os.path.dirname(__file__)
PT_VOCAB_PATH = os.path.join(current_path, '../../datasets/pt_vocab.txt')
ES_VOCAB_PATH = os.path.join(current_path, '../../datasets/es_vocab.txt')
DATASET_PATH = os.path.join(
    current_path, '../../datasets/')
MODEL_PATH = os.path.join(
    current_path, '../../models/translator/ted_hrlr_translate_pt_es_converter')

bert_tokenizer_params = dict(lower_case=True)
pt_tokenizer = text.BertTokenizer(
    PT_VOCAB_PATH, **bert_tokenizer_params)
es_tokenizer = text.BertTokenizer(
    ES_VOCAB_PATH, **bert_tokenizer_params)

examples, metadata = tfds.load(
    'ted_hrlr_translate/es_to_pt', data_dir=DATASET_PATH, with_info=True, as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

print('\n')

for es_examples, pt_examples in train_examples.batch(3).take(1):
    for ex in es_examples:
        print(ex.numpy())

print('\n\n')

train_es = train_examples.map(lambda es, pt: es)
train_pt = train_examples.map(lambda es, pt: pt)

# Tokenize the examples -> (batch, word, word-piece)
token_batch = es_tokenizer.tokenize(es_examples)
# Merge the word and word-piece axes -> (batch, tokens)
token_batch = token_batch.merge_dims(-2, -1)

for ex in token_batch.to_list():
    print(ex)

print('\n')

reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
bert_vocab_args = dict(
    # The target vocabulary size
    vocab_size=8000,
    # Reserved tokens that must be included in the vocabulary
    reserved_tokens=reserved_tokens,
    # Arguments for `text.BertTokenizer`
    bert_tokenizer_params=bert_tokenizer_params,
    # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
    learn_params={},
)

pt_vocab = bert_vocab.bert_vocab_from_dataset(
    train_pt.batch(1000).prefetch(2),
    **bert_vocab_args
)

es_vocab = bert_vocab.bert_vocab_from_dataset(
    train_es.batch(1000).prefetch(2),
    **bert_vocab_args
)

# Lookup each token id in the vocabulary.
txt_tokens = tf.gather(es_vocab, token_batch)
# Join with spaces.
# print(tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1))

words = es_tokenizer.detokenize(token_batch)
tf.strings.reduce_join(words, separator=' ', axis=-1)

START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")


def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)


words = es_tokenizer.detokenize(add_start_end(token_batch))
# print(tf.strings.reduce_join(words, separator=' ', axis=-1))


def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result


token_batch = es_tokenizer.tokenize(es_examples).merge_dims(-2, -1)
words = es_tokenizer.detokenize(token_batch)
# print(words)
# print(cleanup_text(reserved_tokens, words).numpy())


class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text(
            encoding='utf-8').splitlines()
        self.vocab = tf.Variable(vocab)

        # Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)


tokenizers = tf.Module()
tokenizers.pt = CustomTokenizer(reserved_tokens, PT_VOCAB_PATH)
tokenizers.es = CustomTokenizer(reserved_tokens, ES_VOCAB_PATH)

tf.saved_model.save(tokenizers, MODEL_PATH)
