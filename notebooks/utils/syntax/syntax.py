import spacy


class SyntaxPreprocessing:

    def __init__(self):
        self.nlp = spacy.load('pt_core_news_sm')
        
    def parse_text(self, text):
        "Retora uma lista de (palavra, dependencia, pai) do texto de entrada"
        
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
        """
        Retorna uma lista de (sujeito, verbo, objeto) para cada raiz
        encontrada do texto de entrada
        """
        
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