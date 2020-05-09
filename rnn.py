from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
nltk.download('punkt')

def frequencia(docs, MAX_FEATURES):
    maxlen = 0
    word_freqs = collections.Counter()
    num_recs = 0

    for i in docs:
        words = nltk.word_tokenize(i.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1

    vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
    word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    return word2index, vocab_size

def extract(word2index, docs, classes):
    MAX_FEATURES = 10000
    MAX_SENTENCE_LENGTH = 900
    num_recs = len(docs)
    index2word = {v:k for k, v in word2index.items()}

    X = np.empty((num_recs, ), dtype=list)
    y = np.zeros((num_recs, ))
    i = 0

    for i in range(len(docs)):
        words = nltk.word_tokenize(docs[i].lower())
        seqs = []
        for word in words:

            if word in word2index.keys():
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        y[i] = int(classes[i])

    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
    return X

def rnn(Xtrain, Xtest, ytrain, ytest, vocab_size, MAX_SENTENCE_LENGTH, lstm,simple):
    fim_val = int(len(ytest) * 0.75)
    EMBEDDING_SIZE = 64#128
    HIDDEN_LAYER_SIZE = 32#64
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_SIZE,
    input_length=MAX_SENTENCE_LENGTH))
    model.add(SpatialDropout1D((0.2)))

    if lstm:
        model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
    if simple:
        model.add(SimpleRNN(HIDDEN_LAYER_SIZE, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam",
    metrics=["accuracy"])
    history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
    validation_data=(Xtest[:fim_val], ytest[:fim_val]))

    score, acc = model.evaluate(Xtest[fim_val:], ytest[fim_val:], batch_size=BATCH_SIZE)
    print("Test score: %.3f, accuracy: %.3f" % (score, acc))
