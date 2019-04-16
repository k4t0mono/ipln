import nltk


class Preprocessing:

    def __init__(self):
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        self.stemmer = nltk.stem.RSLPStemmer()

    def steammize(self, tokens):
        return [ self.stemmer.stem(x) for x in tokens ]
    
    def tokenize_sents(self, txt):
        sents = self.sent_tokenizer.tokenize(txt)
        return sents
    
    def tokenize_words(self, sent):
        tokens = nltk.tokenize.word_tokenize(sent)
        return(tokens)
    
    def tokenize_text(self, txt):
        tokens = []
        
        sents = self.tokenize_sents(txt)
        for s in sents:
            tokens.append(self.tokenize_words(s))
        
        return tokens