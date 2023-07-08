from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, GRU
import numpy as np
import matplotlib.pyplot as plt


T = 8
D = 2
M = 3

X = np.random.rand(1, T, D)


def lstm1():
    input_ = Input(shape=(T, D))
    rnn = LSTM(M, return_state=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    output, h, c = model.predict(X)
    print("o:", output)
    print("h:", h)
    print("c:", c)


def lstm2():
    input_ = Input(shape=(T, D))
    rnn = LSTM(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    output, h, c = model.predict(X)
    print("o:", output)
    print("h:", h)
    print("c:", c)


def gru1():
    input_ = Input(shape=(T, D))
    rnn = GRU(M, return_state=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    output, h = model.predict(X)
    print("o:", output)
    print("h:", h)


def gru2():
    input_ = Input(shape=(T, D))
    rnn = GRU(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    output, h = model.predict(X)
    print("o:", output)
    print("h:", h)


lstm1()
lstm2()
gru1()
gru2()
