import spacy


class MorphosyntaxProcessing:

    def __init__(self):
        self.nlp = spacy.load('pt_core_news_sm')

    def pos_tag(self, text):
        "Retorna uma lista de (palavra, pos_tag) do texto de entrada"
        
        doc = self.nlp(text)

        tokens = []
        for t in doc:
            tokens.append((t.text, t.pos_))

        return tokens