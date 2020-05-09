#sem neurais

from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
import random

from redes_neurais import frequencia, extract, rnn

nltk.download('punkt')

docs = open("docs.txt",encoding="utf-8").read().split("\n")[0:1000]
classes = open("classes.txt",encoding="utf-8").read().split("\n")[0:1000]

MAX_FEATURES = 10000
MAX_SENTENCE_LENGTH = 900

tam = int(len(docs) / 5)
results = []
kfold = 5
  
results_naive_bayes = []
results_decision_tree = []
results_logistic_regression = []
results_rnn = []
results_lstm = []

for i in range(kfold):
  inicio = i * tam
  fim = (i + 1) * tam
  fim_val = int(tam*0.75)

  Xtrain = docs[:inicio]+docs[fim:]
  Xtest = docs[inicio:fim]
  ytrain = classes[:inicio]+classes[fim:]
  ytest = classes[inicio:fim]
  Xtest1 = Xtest[fim_val:]
  ytest1 = ytest[fim_val:]
  
  vectorizer = feature_extraction.text.CountVectorizer(binary=True)
  
  
  ################################### REGRESSÃO LOGÍSTICA #################################

  
  Xtrain1 = vectorizer.fit_transform(Xtrain,ytrain)
  Xtest1 = vectorizer.transform(Xtest1)


  classificador = LogisticRegression()
  classificador.fit(Xtrain1,ytrain)
  results_logistic_regression.append(classificador.score(Xtest1,ytest1))

  ################################### NAIVE BAYES  #######################################

  classificador = naive_bayes.MultinomialNB()
  classificador.fit(Xtrain1,ytrain)
  results_naive_bayes.append(classificador.score(Xtest1,ytest1))


  ################################### DECISION TREE  #######################################

  classificador = DecisionTreeClassifier()
  classificador.fit(Xtrain1,ytrain)
  results_decision_tree.append(classificador.score(Xtest1,ytest1))

  ########################## RNN ###########################

  word2index, vocab_size = frequencia(docs, MAX_FEATURES)#  TREINO
  Xtrain = extract(word2index,Xtrain, ytrain)
  Xtest = extract(word2index,Xtest, ytest)

  ## LSMT

  results_lstm.append(rnn(Xtrain, Xtest, ytrain, ytest,vocab_size, MAX_SENTENCE_LENGTH, True, False))

  ## SIMPLE

  results_rnn.append(rnn(Xtrain, Xtest, ytrain, ytest,vocab_size, MAX_SENTENCE_LENGTH,False, True))


results_rnn.append(sum(results_rnn)/kfold)
results_lstm.append(sum(results_lstm)/kfold)
results_naive_bayes.append(sum(results_naive_bayes)/kfold)
results_decision_tree.append(sum(results_decision_tree)/kfold)
results_logistic_regression.append(sum(results_logistic_regression)/kfold)

arquivo = open("resultados.txt","a")


arquivo.writelines("Naive Bayes\n")
results_naive_bayes = [str(i)+ " " for i in results_naive_bayes]
arquivo.writelines(results_naive_bayes[:-1])
arquivo.writelines("Média: "+results_naive_bayes[-1])
arquivo.writelines("\n")


arquivo.writelines("Decision Tree\n")
results_decision_tree = [str(i)+ " " for i in results_decision_tree]
arquivo.writelines(results_decision_tree[:-1])
arquivo.writelines("Média: "+results_decision_tree[-1])
arquivo.writelines("\n")



arquivo.writelines("Logistic Regression\n")
results_logistic_regression = [str(i)+ " " for i in results_logistic_regression]
arquivo.writelines(results_logistic_regression[:-1])
arquivo.writelines("Média: "+results_logistic_regression[-1])
arquivo.writelines("\n")

arquivo.writelines("RNN\n")
results_rnn = [str(i)+ " " for i in results_rnn]
arquivo.writelines(results_rnn[:-1])
arquivo.writelines("Média: "+results_rnn[-1])
arquivo.writelines("\n")


arquivo.writelines("LSTM\n")
results_lstm = [str(i)+ " " for i in results_lstm]
arquivo.writelines(results_lstm[:-1])
arquivo.writelines("Média: "+results_lstm[-1])
arquivo.writelines("\n")


print(results_naive_bayes)
print(results_decision_tree)
print(results_logistic_regression)
print(results_lstm)
print(results_rnn)
arquivo.close()