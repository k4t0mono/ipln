import nltk
import unidecode
import string
import spacy


class LexicalProcessing:

    def __init__(self):
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        self.stemmer = nltk.stem.RSLPStemmer()
        self.stopwords = nltk.corpus.stopwords.words('portuguese')
        self.nlp = spacy.load('pt_core_news_sm')

    def remove_accents(self, text):
        "Remove os acentos do texto de entrada"
        
        return unidecode.unidecode(text)

    def remove_punctuation(self, text):
        "Remove a pontuação do texto de entrada"
        
        return text.translate(str.maketrans('','',string.punctuation))

    def tokenize_sentences(self, text):
        "Transforma o texto de entrada em uma lista de sentenças"
        
        sentences = self.sent_tokenizer.tokenize(text)
        return sentences

    def tokenize_words(self, text):
        "Transforma o texto de entrada em uma lista de palavras"
        
        tokens = nltk.tokenize.word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        "Dado uma lista de tokens remove as stopwords"

        t = [ x for x in tokens if x not in self.stopwords ]
        return t

    def lemmatize_sentence(self, text):
        "Dado um texto, retornar o texto lemmatizado"

        doc = self.nlp(text)
        lemmas = [ x.lemma_ for x in doc ]
        return lemmas

    def lemmatize_word(self, word):
        "Retorna o lemma de uma palavra de entrada"
        
        return self.nlp(word)[0].lemma_

    def stemmize(self, tokens):
        "Retorna uma lista de stem da lista de tokens de entrada"
        
        return [self.stemmer.stem(word) for word in tokens]

    def lowercase(self, text):
        "Transforama o texto de entrada em lowercase"
        
        return text.lower()