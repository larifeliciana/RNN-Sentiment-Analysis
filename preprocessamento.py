import spacy
import  random
import nltk
import datetime
#nlp = spacy.load('pt')
nlp = spacy.load('pt_core_news_sm')

stopwords = open("stopwords.txt", encoding="utf-8").read().split("\n")
#pontuação stopwords lowercase
neg = open("app_pt_neg__",encoding="utf-8").read().split("\n")
pos = open("app_pt_pos__",encoding="utf-8").read().split("\n")


def preprocessamento(lista):
    print("preprocessando...")
    lista_preprocessada = []
    inteiro = 0;

    for i in lista:

        inteiro = inteiro+1
        doc = nlp(i)
        new_doc = ""
        for token in doc:
            if not token.is_punct and token.is_alpha and not str(token) in stopwords and len(token) > 1:
                new_doc = new_doc + " "+ str(token).lower()
        lista_preprocessada.append(new_doc)
    return lista_preprocessada

def salvar_lista(endereco, lista):

    arq = open(endereco, "wt", encoding="utf-8")

    for element in lista:
        arq.write(str(element) + '\n')

    arq.close()

nova_pos = preprocessamento(pos)
nova_neg = preprocessamento(neg)

docs = nova_pos+nova_neg
classes = [1]*len(nova_pos) + [0]*len(nova_neg)

z = list(zip(docs, classes))
random.shuffle(z)
docs, classes = zip(*z)

salvar_lista("docs.txt", docs)
salvar_lista("classes.txt", classes)
