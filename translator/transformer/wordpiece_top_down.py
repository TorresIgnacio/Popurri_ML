import os
import re
import tensorflow as tf
import tensorflow_text as tf_text
import pandas as pd
import numpy as np

# The Algorithm: https://www.tensorflow.org/text/guide/subwords_tokenizer#algorithm

# Preparation
current_path = os.path.dirname(__file__)
filepath = os.path.join(current_path, '../../datasets/spa.txt')
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    inputs = []
    targets = []
    for line in lines:
        input_text, target_text, _ = line.split('\t')
        inputs.append(input_text)
        targets.append(target_text)

corpus = []
for text in inputs:
    # Split accented chars
    text = text.lower()
    # Keep space, a to z, and some punctuation.
    text = re.sub('[^ a-z.=!,¿]', '', text)
    # Add spaces around punctuation.
    text = re.sub('[.?!,¿]', r' \0 ', text)
    # Strip whitespace.
    text = text.strip()
    corpus.append(text)

# First Iteration
# Step 1: Iterate over every word and count pair in the input, denoted as (w, cc)

print('Iteration 1')
vocab = {}
for lines in corpus:
    words = lines.split()
    for word in words:
        value = vocab.get(word)
        if value is None:
            vocab[word] = 1
        else:
            vocab[word] = value + 1

df = pd.DataFrame.from_dict(vocab, orient='index')
df.drop(index=['\0'], axis=1, inplace=True)

# Step 2: For each word w, generate every substring, denoted as s. E.g., for the word human, we generate {h, hu, hum, huma, human, ##u, ##um, ##uma, ##uman, ##m, ##ma, ##man, #a, ##an, ##n}
# Step 3: Maintain a substring-to-count hash map, and increment the count of each s by c. E.g., if we have (human, 113) and (humas, 3) in our input, the count of s = huma will be 113+3=116

vocab_2 = {}
for key in df.index:
    key_length = len(key)
    for left_window in range(key_length):
        for right_window in range(left_window, key_length):
            if left_window == 0:
                substring = key[left_window:right_window + 1]
            else:
                substring = '##'+key[left_window:right_window + 1]
            substring_count = vocab_2.get(substring)
            word_count = df.loc[key][0]
            if substring_count is None:
                vocab_2[substring] = word_count
            else:
                vocab_2[substring] = substring_count + word_count

vocab_df = pd.DataFrame(
    list(vocab_2.items()), columns=['Word', 'Count'])

# *Step 4: Once we've collected the counts of every substring, iterate over the (s, c) pairs starting with the longest s first.
# *Step 5: Keep any s that has a c > T. E.g., if T = 100 and we have (pers, 231); (dogs, 259); (##rint; 76), then we would keep pers and dogs.
# *Step 6: When an s is kept, subtract off its count from all of its prefixes. This is the reason for sorting all of the s by length in step 4. This is a critical part of the algorithm, because otherwise words would be double counted. For example, let's say that we've kept human and we get to (huma, 116). We know that 113 of those 116 came from human, and 3 came from humas. However, now that human is in our vocabulary, we know we will never segment human into huma ##n. So once human has been kept, then huma only has an effective count of 3

sorted_df = vocab_df.sort_values(by='Word', key=(
    lambda x: x.str.len()), ascending=False)
T = 100
vocab_3 = {}

# Step 4
for word in sorted_df['Word']:
    word_count = vocab_2[word]
    # Step 5
    if word_count < T:
        continue
    vocab_3[word] = word_count

    # Step 6
    for i in range(len(word)):
        subword_count = vocab_2.get(word[:i+1])
        if subword_count is not None:
            vocab_2[word[:i+1]] -= word_count


# Clear words with count = 0
vocab_3 = {k: v for k, v in vocab_3.items() if v != 0}


# Iteration 2
# Subsequent iterations are identical to the first, with one important distinction: In step 2, instead of considering every substring, we apply the WordPiece tokenization algorithm using the vocabulary from the previous iteration, and only consider substrings which start on a split point.
# We will only consider substrings that start at a segmentation point. We will still consider every possible end position.

# ***Applying WordPiece***
# For example, consider segmenting the word undeniable.

# We first lookup undeniable in our WordPiece dictionary, and if it's present, we're done. If not, we decrement the end point by one character, and repeat, e.g., undeniabl.

# Eventually, we will either find a subtoken in our vocabulary, or get down to a single character subtoken. (In general, we assume that every character is in our vocabulary, although this might not be the case for rare Unicode characters. If we encounter a rare Unicode character that's not in the vocabulary we simply map the entire word to <unk>).

# In this case, we find un in our vocabulary. So that's our first word piece. Then we jump to the end of un and repeat the processing, e.g., try to find ##deniable, then ##deniabl, etc. This is repeated until we've segmented the entire word.

print('Iteration 2')

vocab_4 = {}
vocab_4['<unk>'] = 0
for word in vocab.keys():
    word_count = vocab_3.get(word)
    if word_count is not None:
        vocab_4[word] = word_count
        continue

    # Apply WordPiece
    left = 0
    while (True):
        for right in range(len(word), left, -1):
            substring = word[left:right] if left == 0 else '##' + \
                word[left:right]
            word_count = vocab_3.get(substring)
            if word_count is not None:
                previous_count = vocab_4.get(substring)
                vocab_4[substring] = word_count if previous_count is None else previous_count + word_count
                left = right
                break

        if left == right - 1:
            vocab_4['<unk>'] += 1
            left = right
        if left == len(word):
            break


def tokenize(word, vocab):
    tokenized = ''
    if vocab.get(word) is not None:
        return word
    # Apply WordPiece
    left = 0
    while (True):
        for right in range(len(word), left, -1):
            substring = word[left:right] if left == 0 else '##' + \
                word[left:right]
            if vocab.get(substring) is not None:
                tokenized = ''.join([tokenized, substring])
                left = right
                break
        if left == right - 1:
            tokenized = ''.join([tokenized, '<unk>'])
            left = right
        if left == len(word):
            break
    return tokenized
