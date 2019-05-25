import nltk
import unidecode
import string
import spacy


class Preprocessing:

    def __init__(self):
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        self.stemmer = nltk.stem.RSLPStemmer()
        self.nlp = spacy.load('pt_core_news_sm')

    def remove_accents(self, text):
        return unidecode.unidecode(text)

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('','',string.punctuation))

    def tokenize_sentences(self, text):
        sentences = self.sent_tokenizer.tokenize(text)
        return sentences

    def tokenize_words(self, text):
        tokens = nltk.tokenize.word_tokenize(text)
        return tokens

    def lemmatize(self, text):
        return self.nlp(text)[0].lemma_

    def stemmize(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    def lowercase(self, text):
        return text.lower()
    
    def pos_tag(self, text):
        doc = self.nlp(text)

        tokens = []
        for t in doc:
            tokens.append((t.text, t.pos_))

        return tokens
    
    def parse_text(self, text):
        doc = self.nlp(text)

        tokens = []
        for t in doc:
            if t.dep_ == 'ROOT':
                h = None
            else:
                h = t.head.text

            tokens.append((t.text, t.dep_, h))

        return tokens
    
    def get_svo(self, txt):
        doc = self.nlp(txt)

        roots = []
        for t in doc:
            if t.dep_ == 'ROOT':
                roots.append(t)

        result = []
        for r in roots:
            if r.pos_ != 'VERB':
                continue

            verb = r.text
            subj = None
            dobj = None
            iobj = None

            children = [ c for c in r.children ]

            for c in children:
                # print(c.text, c.lemma_, c.pos_, c.dep_)
                if 'subj' in c.dep_:
                    subj = c.text

                if c.dep_ in [ 'obj',  ]:
                    dobj = c.text

                if c.dep_ in [ 'obl', 'xcomp' ]:
                    iobj = c.text
            
            # print(dobj, iobj)
            if dobj:
                result.append((subj, verb, dobj))
            elif iobj:
                result.append((subj, verb, iobj))
            else:
                result.append((subj, verb, None))
        
        # print(x)
        return result