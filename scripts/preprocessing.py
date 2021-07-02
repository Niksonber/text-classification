import re
import string

import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk import download

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from typing import List
from abc import ABC, abstractmethod

download('punkt')
download('stopwords')

"""! Mother class of all preprocessors"""
class Preprocessor:
    def __init__(self) -> None:
        pass
    
    """! process input prototype """
    @abstractmethod
    def processSample(self, x):
        pass
    
    """! """
    def process(self, x:list) -> list:
        return [self.processSample(sample) for sample in x]

"""! Split text in words"""
class Tokenize(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

    """! Split text in words
        @param x text
        @return words lists"""
    def processSample(self, sample:str) -> list:
        return word_tokenize(sample)

"""! Normalize inputs, transforming in lower case, removing punctuation and non-alphanumeric words"""
class Normalize(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

    """! 
        @param x words list
        @return words lists of low case, alphanumeric words without punctuation an"""
    def processSample(self, sample:list) -> list:
        # used to remove punctuation
        punctuation_re = re.compile('[%s]'% re.escape(string.punctuation))
        # lower case
        sample = [word.lower() for word in sample]
        # remove punctuation
        sample = [punctuation_re.sub('', word) for word in sample]
        # remove all words that have non-alphanumeric characters
        sample = [word for word in sample if word.isalpha()]
    
        return sample

"""! Remove stop words"""
class RemoveStopWords(Preprocessor):
    def __init__(self, language = 'english') -> None:
        super().__init__()
        self._language = language
    
    """! Remove stop words
        @param x words list
        @return words list without non-significative words"""
    def processSample(self, sample:list) -> list:
        stop_words = set(stopwords.words(self._language))
        return [word for word in sample if not word in stop_words]

"""!  Stem words """
class Steamer(Preprocessor):
    def __init__(self, language = 'english') -> None:
        super().__init__()
        self._porter = SnowballStemmer(language)
        self._language = language
    """! Stem words
    @param x words list
    @return words list without non-significative variations"""
    def processSample(self, sample:list) -> list:
        stop_words = set(stopwords.words(self._language))
        return [self._porter.stem(word) for word in sample]

"""! Break sequence length in in mean + 2 * std """
class BreakSequence(Preprocessor):
    def __init__(self, plot:bool = False) -> None:
        super().__init__()
        self._plot = plot
    

    """! Break sequence in indicated size
    @param x list of words list
    @return list od words list with croped length"""
    def process(self, x:list) -> list:
        lengths = [len(sample) for sample in x]
        std = np.std(lengths)
        mean = np.mean(lengths)
        # 95% interval (assuming gaussian distribuition)
        recommended = int(mean + 2 * std)

        # Histogram
        if self._plot:
            plt.hist(lengths, bins=np.arange(min(lengths), max(lengths), 20))
            plt.title("Comprimento dos textos pré-processados (em número de palavras)")
            plt.ylabel("Numero de amostras")
            plt.xlabel("Comprimento do texto (rm palavras)")
            plt.axvline(recommended, color = 'r')
            plt.show()

        print({"min" : np.min(lengths), "max": max(lengths), "mean": mean, "std": std, "recomended": recommended})
        return [ sample[:recommended] for sample in x ]
    
"""! Dry rare words"""
class Dry(Preprocessor):
    def __init__(self, min_occ:int = 4) -> None:
        super().__init__()
        self._wordCount = {}
        self._words = []
        self.min_occ = min_occ
    
    """! Generate counter of each word
    @param data list of preprocessed list of words
    @return dictionary of one hot encoder"""
    def generate(self, data) -> dict:
        for sample in data:
            for word in sample:
                if word in self._wordCount:
                    self._wordCount[word] += 1
                else:
                    self._wordCount[word] = 1

        #filter rare words
        self._words = [key for key, value in self._wordCount.items() if value > self.min_occ]
        return self._wordCount

    """! Remove rare words
    @param x words list
    @return words list without rare words"""
    def processSample(self, sample:list) -> list:
        return [word for word in sample if word in self._words]

    """! Remove rare words
    @param x list of words list
    @return list od words list without rare words"""
    def process(self, x: list) -> list:
        self.generate(x)
        return super().process(x)

"""!  Join words list in text"""
class Join(Preprocessor):
    
    """! Join words list in text
    @param x words list
    @return text joined with space"""
    def processSample(self, sample:list) -> str:
        return [" ".join(words) for words in sample]

"""! Encode samples and add padding"""
class Words2index:

    """! Encode samples and add padding
        @param x texx list
        @return sequence of int list """
    def process(self, x: list) -> list:
        tokenizer=Tokenizer()
        tokenizer.fit_on_texts(x)
        x=tokenizer.texts_to_sequences(x)
        return pad_sequences(x, maxlen=427, padding="post")

if __name__ == "__main__":
    pass