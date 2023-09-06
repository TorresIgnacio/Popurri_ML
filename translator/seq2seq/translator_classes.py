import tensorflow as tf
import tensorflow_text as tf_text
import einops
import pathlib
import numpy as np
import matplotlib.pyplot as plt

tf.keras.utils.get_custom_objects().clear()


@tf.keras.utils.register_keras_serializable(package="Translator")
class ShapeChecker():
    def __init__(self):
        # Keep a cache of every axis-name seen
        self.shapes = {}

    def get_config(self):
        config = {'shapes': self.shapes}
        return config

    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return

        parsed = einops.parse_shape(tensor, names)

        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)

            if (broadcast and new_dim == 1):
                continue

            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                 f"    found: {new_dim}\n"
                                 f"    expected: {old_dim}\n")


def load_data(path):
    text = path.read_text(encoding='utf-8')

    lines = text.splitlines()

    inputs = []
    targets = []

    for line in lines:
        input_text, target_text, _ = line.split('\t')
        inputs.append(input_text)
        targets.append(target_text)

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


@tf.keras.utils.register_keras_serializable(package="Translator")
def tf_lower_and_split_punct(text):
    # Split accented chars
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and some punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

    return text


@tf.keras.utils.register_keras_serializable(package="Translator")
class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, units, mask_zero=True)

        # RNN layer
        self.rnn = tf.keras.layers.Bidirectional(merge_mode='sum', layer=tf.keras.layers.GRU(
            units, return_sequences=True, recurrent_initializer='glorot_uniform'))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "text_processor": self.text_processor,
                "units": self.units,
                "embedding": self.embedding,
                "rnn": self.rnn
            }
        )
        return config

    def call(self, x):
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch s')

        x = self.embedding(x)
        shape_checker(x, 'batch s units')

        x = self.rnn(x)
        shape_checker(x, 'batch s units')

        return x

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        inputs = self.text_processor(texts).to_tensor()
        inputs = self(inputs)
        return inputs


@tf.keras.utils.register_keras_serializable(package="Translator")
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.units = units
        self.mha = tf.keras.layers.MultiHeadAttention(
            key_dim=units, num_heads=1, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "units": self.units,
                "mha": self.mha,
                "layernorm": self.layernorm,
                "add": self.add
            }
        )
        return config

    def call(self, x, context):
        shape_checker = ShapeChecker()

        shape_checker(x, 'batch t units')
        shape_checker(context, 'batch s units')

        attn_output, attn_scores = self.mha(
            query=x, value=context, return_attention_scores=True)

        shape_checker(x, 'batch t units')
        shape_checker(attn_scores, 'batch heads t s')

        # Cache attention scores
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        shape_checker(attn_scores, 'batch t s')
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


@tf.keras.utils.register_keras_serializable(package="Translator")
class Decoder(tf.keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=self.text_processor.get_vocabulary(), mask_token='', oov_token='[UNK]')
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=self.text_processor.get_vocabulary(), mask_token='', oov_token='[UNK]', invert=True)
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')

        self.units = units

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, units, mask_zero=True)

        # RNN layer
        self.rnn = tf.keras.layers.GRU(
            units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        # Attention layer
        self.attention = CrossAttention(units)

        # Output layer
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def get_config(self):
        config = super().get_config()
        # Update the config with the custom layer's parameters
        config.update(
            {
                "text_processor": self.text_processor,
                "units": self.units,
                "word_to_id": self.word_to_id,
                "id_to_word": self.id_to_word,
                "embedding": self.embedding,
                "rnn": self.rnn,
                "attention": self.attention,
                "output_layer": self.output_layer
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["attention"] = tf.keras.layers.deserialize(config["Attention"])
        return cls(**config)


@Decoder.add_method
def call(self, inputs, x, state=None, return_state=False):
    shape_checker = ShapeChecker()
    shape_checker(x, 'batch t')
    shape_checker(inputs, 'batch s units')

    x = self.embedding(x)
    shape_checker(x, 'batch t units')

    x, state = self.rnn(x, initial_state=state)
    shape_checker(x, 'batch t units')

    x = self.attention(x, inputs)
    self.last_attention_weights = self.attention.last_attention_weights
    shape_checker(x, 'batch t units')
    shape_checker(self.last_attention_weights, 'batch t s')

    logits = self.output_layer(x)
    shape_checker(logits, 'batch t target_vocab_size')

    if return_state:
        return logits, state

    return logits


@Decoder.add_method
def get_initial_state(self, inputs):
    batch_size = tf.shape(inputs)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]


@Decoder.add_method
def tokens_to_text(self, tokens):
    words = self.id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
    result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
    return result


@Decoder.add_method
def get_next_token(self, inputs, next_token, done, state, temperature=0.0):
    logits, state = self(
        inputs, next_token,
        state=state,
        return_state=True)

    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits[:, -1, :]/temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    # If a sequence produces an `end_token`, set it `done`
    done = done | (next_token == self.end_token)
    # Once a sequence is done it only produces 0-padding.
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

    return next_token, done, state


@tf.keras.utils.register_keras_serializable(package="Translator")
class Translator(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units,
                 input_text_processor,
                 target_text_processor):
        super().__init__()
        self.units = units
        self.input_text_processor = input_text_processor
        self.target_text_processor = target_text_processor
        # Build the encoder and decoder
        self.encoder = Encoder(input_text_processor, units)
        self.decoder = Decoder(target_text_processor, units)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "input_text_processor": self.input_text_processor,
                "target_text_processor": self.target_text_processor,
                "encoder": self.encoder,
                "decoder": self.decoder
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        # Note that you can also use `keras.saving.deserialize_keras_object` here
        config["encoder"] = tf.keras.layers.deserialize(config["Encoder"])
        config["decoder"] = tf.keras.layers.deserialize(config["Decoder"])
        return cls(**config)

    def call(self, inputs):
        input_tokens, target_in = inputs
        enc_output = self.encoder(input_tokens)
        logits = self.decoder(enc_output, target_in)
        return logits


def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)


@Translator.add_method
def translate(self,
              texts, *,
              max_length=50,
              temperature=0.0):
    # Process the input texts
    enc_output = self.encoder.convert_input(texts)
    batch_size = tf.shape(texts)[0]

    # Setup the loop inputs
    tokens = []
    attention_weights = []
    next_token, done, state = self.decoder.get_initial_state(enc_output)

    for _ in range(max_length):
        # Generate the next token
        next_token, done, state = self.decoder.get_next_token(
            enc_output, next_token, done,  state, temperature)

        # Collect the generated tokens
        tokens.append(next_token)
        attention_weights.append(self.decoder.last_attention_weights)

        if tf.executing_eagerly() and tf.reduce_all(done):
            break

    # Stack the lists of tokens and attention weights.
    tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)
    self.last_attention_weights = tf.concat(
        attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)

    result = self.decoder.tokens_to_text(tokens)
    return result


@Translator.add_method
def plot_attention(self, text, **kwargs):
    assert isinstance(text, str)
    output = self.translate([text], **kwargs)
    output = output[0].numpy().decode()

    attention = self.last_attention_weights[0]

    context = tf_lower_and_split_punct(text)
    context = context.numpy().decode().split()

    output = tf_lower_and_split_punct(output)
    output = output.numpy().decode().split()[1:]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    ax.matshow(attention, cmap='viridis', vmin=0.0)

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + context, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + output, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xlabel('Input text')
    ax.set_ylabel('Output text')


class Export(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, inputs):
        return self.model.translate(inputs)
