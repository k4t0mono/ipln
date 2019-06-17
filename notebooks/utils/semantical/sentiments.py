import pickle
import gzip
import pandas as pd
import numpy
from sklearn.neural_network import MLPClassifier
from gensim.models import KeyedVectors
from utils.lexical import LexicalProcessing
import os

class SentimentAnalysis:

    def __init__(self):
        self.LP = LexicalProcessing()
        print(os.getcwd())
        print(os.listdir('../models'))
        with gzip.open('../models/model_mlp.pkl.gz', 'rb') as f:
            self.model = pickle.loads(f.read())

        self.w2v = KeyedVectors.load_word2vec_format('../models/cbow_s50.txt')

    def _calc_vec(self, tokens):
        vecs = []
        for w in tokens:
            try:
                vecs.append(self.w2v[w])
            except:
                pass

        if not vecs:
            return numpy.zeros((50, ))
        
        return numpy.average(vecs, axis=0)

    def sentiment_score(self, text):
        "Prediz o sentimento de uma frase de 0-5"

        a = self.LP.lowercase(text)
        a = self.LP.remove_punctuation(a)
        a = self.LP.lemmatize_sentence(a)
        vec = self._calc_vec(a)
        
        return self.model.predict([vec])[0]