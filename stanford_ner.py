'''
A sample code usage of the python package stanfordcorenlp to access a Stanford CoreNLP server.
Written as part of the blog post: https://www.khalidalnajjar.com/how-to-setup-and-use-stanford-corenlp-server-with-python/ 
'''

from stanfordcorenlp import StanfordCoreNLP
import logging
import json

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

if __name__ == '__main__':
    sNLP = StanfordNLP()
    #text = 'Beyonce lives in Los Angeles. I work in New York City.'

    #text = "Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.  Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pass to rookie wide receiver Chaz Schilens.  The Texans would respond with fullback Vonta Leach getting a 1-yard touchdown run, yet the Raiders would answer with kicker Sebastian Janikowski getting a 33-yard and a 30-yard field goal.  Houston would tie the game in the second quarter with kicker Kris Brown getting a 53-yard and a 24-yard field goal. Oakland would take the lead in the third quarter with wide receiver Johnnie Lee Higgins catching a 29-yard touchdown pass from Russell, followed up by an 80-yard punt return for a touchdown.  The Texans tried to rally in the fourth quarter as Brown nailed a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt."
    text = "How many field goals did Kris Brown kick?"
    annotations = sNLP.annotate(text)

    FLAG_NER = "tp@ckl"
    FLAG_SENTENCE = "tp#ckl"
    
    annotated_text = ""
    
    for sentence in annotations['sentences']:
        # tokens = list of dictionaries, each dictionary = token (word)
        for token in sentence['tokens']:
            if token['ner'] == 'O':
                annotated_text += token['word'] 
            else:
                annotated_text += token['word'] + FLAG_NER + token['ner'] + FLAG_NER 

            # append space, unless token is last token of the sentence
            annotated_text += " " if token['index'] != len(sentence['tokens']) else ""
        
        # append space, unless sentence is last sentence of the text
        annotated_text += FLAG_SENTENCE + " " if sentence['index'] != len(annotations['sentences']) else ""

        # print("Sentence index:", sentence['index'])
        # print("Entities:", sentence['entitymentions'])
        # print("Tokens:", sentence['tokens'])

    print(annotated_text)