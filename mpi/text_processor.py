import numpy as np
import pandas as pd
import unidecode
import unicodedata
import re
import spacy
import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer
from spellchecker import SpellChecker
import spacy_spanish_lemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
import num2words

spell = SpellChecker(language='es')
stemmer = SnowballStemmer('spanish')
nltk.download('wordnet')
nltk.download('stopwords')
spanish_stopwords = list(set(stopwords.words('spanish')))

class TextProcessor(BaseEstimator, TransformerMixin): 
    def remove_numbers(self, sentence):
        return re.sub("\d+", " ", sentence)
    
    def stopwords(self, sentence):
        palabras = sentence.split(' ')
        frase = []
        for p in palabras:
            if p not in spanish_stopwords:
                frase.append(p)
        return ' '.join(frase)
    
    def remove_punctuation(self, sentence):
        return re.sub(r'[^\w\s]', ' ', sentence)
    
    def remove_accents(self, sentence):
        return unidecode.unidecode(str(sentence))
    
    def stemming(self, sentence):
        palabras = sentence.split(' ')
        frase = []
        for p in palabras:
            new_p = stemmer.stem(p)
            frase.append(new_p)
        return ' '.join(frase)
    
    def remove_multiple_spaces(self, sentence):
        return re.sub(' +', ' ', sentence)
    
    
    def preprocess_text(self, text):
        return self.stemming(
                 self.stopwords(
                    self.remove_multiple_spaces(
                        self.remove_numbers(
                            self.remove_punctuation(
                                self.remove_accents(
                                    text.lower().strip()
                                )
                            )
                        )
                    )
                )
        ).strip()
                #).strip()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, sentence):
        return sentence.apply(self.preprocess_text).values
