# -*- coding: utf-8 -*-
"""redes_neurais.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lLLOQSHG5a0KpbQZSe4SAKs9v4wspZEt
"""

from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_extraction

  

  
def classificadores(Xtrain, Xtest, ytrain, ytest):
  extract = feature_extraction.text.CountVectorizer()
  inicio_val = int(len(ytrain) * 0.75)
    
  Xtrain = Xtrain[:inicio_val]
  ytrain = ytrain[:inicio_val]

  Xtrain = extract.fit_transform(Xtrain)
  print(len(extract.get_feature_names()))
 
  Xtest = extract.transform(Xtest)
  #print("inicio")
 
#  print(len(ytrain))
  classificador = naive_bayes.MultinomialNB()#RandomForestClassifier()#DecisionTreeClassifier()
  classificador.fit(Xtrain, ytrain)
  acc = classificador.score(Xtest, ytest)
  arquivo = open("DT_results.txt", "a")  
  arquivo.writelines("DT ACC: ")
  arquivo.writelines(str(acc))
  print(acc)
  arquivo.writelines("\n*")
 
  arquivo.close()
  return acc



from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers import TimeDistributed, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.optimizers import SGD, Adam, RMSprop
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
#MAX_FEATURES = 5000
#MAX_SENTENCE_LENGTH = 900
def frequencia(tudo, MAX_FEATURES):
    maxlen = 0
    word_freqs = collections.Counter()
    num_recs = 0

    for i in tudo:
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

def extract(word2index, tudo, tudo_classes, MAX_SENTENCE_LENGTH):
    num_recs = len(tudo)
    index2word = {v:k for k, v in word2index.items()}

    X = np.empty((num_recs, ), dtype=list)
    y = np.zeros((num_recs, ))
    i = 0

    for i in range(len(tudo)):
        words = nltk.word_tokenize(tudo[i].lower())
        seqs = []
        for word in words:

            if word in word2index.keys():
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs
        y[i] = int(tudo_classes[i])

    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
    return X#, y
#    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
 #   return Xtrain, Xtest, ytrain, ytest

def rnn(Xtrain, Xtest, ytrain, ytest, vocab_size, MAX_SENTENCE_LENGTH, lstm,simple, HIDDEN_LAYER_SIZE, BATCH_SIZE, l1, l2):
  
    print("########################## LSTM ##########################")
  
    inicio_val = int(len(ytrain) * 0.75)
    Xval = Xtrain[inicio_val:]
    yval = ytrain[inicio_val:]
    Xtrain = Xtrain[:inicio_val]
    ytrain = ytrain[:inicio_val]
    
    EMBEDDING_SIZE = 128
    NUM_EPOCHS = 5
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_SIZE,
    input_length=MAX_SENTENCE_LENGTH))
    
    model.add(SpatialDropout1D((0.2)))

    if lstm:
          #model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2),merge_mode='concat'))
          model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
  

         # model.add(TimeDistributed(Dense(1,activation='relu')))
         # model = Flatten()
    if simple:
        model.add(SimpleRNN(HIDDEN_LAYER_SIZE, return_sequences = True))
        model.add(SimpleRNN(HIDDEN_LAYER_SIZE))

    if l1 and not l2:
      model.add(Dense(1, kernel_regularizer=regularizers.l1(0.01)))
    if l2 and not l1:
      model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
    if l1 and l2:
   #   model.add(Dense(100, activation='sigmoid'))
   #   model.add(Dropout(0.2))
      model.add(Dense(1, kernel_regularizer=regularizers.L1L2(0.01)))

    if not l1 and not l2:
      model.add(Dense(1))
      
 
    model.add(Activation("sigmoid"))
  #  model.summary()
    #input()
  
    optim = "adam"
    model.compile(loss="binary_crossentropy", optimizer=optim,
    metrics=["accuracy"])
    history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, verbose = 1, epochs=NUM_EPOCHS,
    validation_data=(Xval, yval ))
    
    reg = ""
    if l1:
      reg = "L1 "
    if l2:
      reg = reg +"L2 "
    if not l1 and not l2:
      reg = "None "
    if lstm:
      rede = "LSTM"
    else: rede = "RNN"
    if lstm:
      arquivo = open("LSTM.txt", "a")
    else: arquivo = open("RNN.txt", "a")
    arquivo.writelines(rede+" - " +reg+ str(EMBEDDING_SIZE)+ " "+ str(HIDDEN_LAYER_SIZE)+ " " + str(BATCH_SIZE)+"\n")
    arquivo.writelines("ACC_TRAIN: ")
    arquivo.writelines([str(i)+" " for i in history.history['acc']])
    arquivo.writelines("\n")
  
  
    arquivo.writelines("LOSS_TRAIN: ")
    arquivo.writelines([str(i)+" " for i in history.history['loss']])
    arquivo.writelines("\n")
  
  
    arquivo.writelines("VAL_ACC: ")
    arquivo.writelines([str(i)+" " for i in history.history['val_acc']])
    arquivo.writelines("\n")
    
    arquivo.writelines("VAL_LOSS: ")
    arquivo.writelines([str(i)+" " for i in history.history['val_loss']])
    arquivo.writelines("\n")
  
  
    #['acc', 'loss', 'val_acc', 'val_loss']

    
    score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
    print(score)
    print(acc)
    arquivo.writelines("LOSS_TEST:")
    arquivo.writelines(str(score))
    arquivo.writelines("\n")
    
    arquivo.writelines("ACC_TEST: ")
    arquivo.writelines(str(acc))
    arquivo.writelines("\n")
    arquivo.close()

from sklearn.model_selection import train_test_split
import random
import numpy as np
nltk.download('punkt')

#neg = open("steam_pt_neg_preprocessada",encoding="utf-8").read().split("\n")[0:1000]
#pos = open("steam_pt_pos_preprocessada",encoding="utf-8").read().split("\n")[0:1000]
#neg_classes = [0]*len(neg)
#pos_classes = [1]*len(pos)


#tudo = neg+pos
#tudo_classes = neg_classes+pos_classes

tudo = open("tudo_.txt", encoding = "utf-8").read().split("\n")
tudo_classes = open("tudo_classes_.txt",encoding = "utf-8").read().split("\n")
tudo_classes = [int(i) for i in tudo_classes]

def experimento(layer, batch, lstm, l1, l2):
  MAX_FEATURES = 10000
  MAX_SENTENCE_LENGTH = 900

  results = []
  kfold = 10
  total = 0
  tam = int(len(tudo) / kfold)
  total = 0
  for i in range(kfold):

    print("fold " + str(i))

    inicio = i * tam
    fim = (i + 1) * tam

    Xtrain = tudo[:inicio]+tudo[fim:]
    Xtest = tudo[inicio:fim]
    ytrain = tudo_classes[:inicio]+tudo_classes[fim:]
    ytest = tudo_classes[inicio:fim]

    results_rnn = []
    results_lstm = []
    print(classificadores(Xtrain, Xtest, ytrain, ytest))

    ########################## RNN ###########################
    word2index, vocab_size = frequencia(Xtrain, MAX_FEATURES)#  TREINO
  
    Xtrain = extract(word2index,Xtrain, ytrain, MAX_SENTENCE_LENGTH)
    Xtest = extract(word2index,Xtest, ytest, MAX_SENTENCE_LENGTH)

    ## LSMT


    np.random.seed(54321 + i*400)
   # total = total + 
    
    if lstm:
      results_lstm.append(rnn(Xtrain, Xtest, ytrain, ytest, vocab_size, MAX_SENTENCE_LENGTH, True, False, layer, batch, l1=l1, l2=l2))
    else:
  ## SIMPLE
  # np.random.seed(54321 + i*400)
      results_rnn.append(rnn(Xtrain, Xtest, ytrain, ytest,vocab_size, MAX_SENTENCE_LENGTH,False, True, layer, batch,l1=l1, l2=l2))

#  print(total/10)
#results_rnn.append(sum(results_rnn)/kfold)
#results_lstm.append(sum(results_lstm)/kfold)

"""10000  900 | RODAR COM TUDO

5 ÉPOCAS | TESTAR MAIS

10 FOLDS 

 REDES NEURAIS: RNN, GRU e LSTM 

ALGORITMOS: ÁRVORE DE DECISÃO, NAIVE BAYES e REGRESSÃO LOGÍSTICA
"""





#experimento(64, 32, False)

#RNN
#experimento(32, 32, False,True,True)
#LSTM
experimento(32, 32, True, True, True)