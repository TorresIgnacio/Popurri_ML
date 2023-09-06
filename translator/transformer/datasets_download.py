import collections
import os
import pathlib
import re
import string
import sys
import tempfile
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

DATA_DIR = '../../datasets/'

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

examples, metadata = tfds.load('ted_hrlr_translate/es_to_pt', data_dir=DATA_DIR, with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

for es, pt in train_examples.take(1):
    print("Portuguese: ", pt.numpy().decode('utf-8'))
    print("Español:   ", es.numpy().decode('utf-8'))

train_es = train_examples.map(lambda es, pt: es)
train_pt = train_examples.map(lambda es, pt: pt)

bert_tokenizer_params = dict(lower_case=True)
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


def write_vocab_file(filepath, vocab):
    with open(filepath, 'w', encoding='utf-8') as f:
        for token in vocab:
            print(token, file=f)


write_vocab_file('../../datasets/pt_vocab.txt', pt_vocab)

es_vocab = bert_vocab.bert_vocab_from_dataset(
    train_es.batch(1000).prefetch(2),
    **bert_vocab_args
)

write_vocab_file('../../datasets/es_vocab.txt', es_vocab)
