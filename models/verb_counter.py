from functools import reduce
import numpy as np
import pandas as pd

from nltk import sent_tokenize, pos_tag, word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class VerbCounter(BaseEstimator, TransformerMixin):

    def count_verbs(self, text):
        # tokenize by sentences
        sentence_list = sent_tokenize(text)

        total = 0
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = pos_tag(word_tokenize(sentence))

            total += reduce(lambda total, word: total + 1 if 'VB' in word[1] else total, pos_tags, 0)

        return total

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        vect_func = np.vectorize(self.count_verbs)
        return pd.DataFrame(vect_func(X))