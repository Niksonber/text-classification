import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk import download
from difflib import get_close_matches

download('punkt')
download('stopwords')


"""! Preprocess text, splitting words, removing ponctuation, crop sequence and stem"""
class Preprocessor:
    def __init__(self) -> None:
        pass

    """! Split words, lower case, remove punctuation, remove stopwords and stem
        @param text target text @param max_len max (optional) length of sequence """
    def preprocess(self, text, max_len = None) -> list:
        # remove spaces
        tokens = word_tokenize(text)

        # convert to lower case
        tokens = [w.lower() for w in tokens]

        # remove punctuation from each word
        re_punc = re.compile('[%s]'% re.escape(string.punctuation))
        stripped = [re_punc.sub('', w) for w in tokens]
       
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        #stem and crop sequence
        porter = SnowballStemmer("english")
        stemmed = [porter.stem(word) for word in words[:max_len]]
        return stemmed

class OneHotEndoder:
    def __init__(self) -> None:
        # Start of string
        self.SOS = "__SOS__"

        #end of string
        self.EOS = "__EOS__"

        # encoder
        self._word2index = {self.SOS:0, self.EOS: 1}
        #number of words
        self._nWords = 2
    
    """! Generate one hot encoder from data
        @param data list of preprocessed list of words
        @return dictionary of one hot encoder"""
    def generate(self, data) -> dict:
        for sample in data:
            for word in sample:
                self.add(word)
        return self._word2index

    """! If not encoded, add to dictionary
        @param word"""
    def add(self, word:str) -> None:
        if word not in self._word2index:
            self._word2index[word] = self._nWords
            self._nWords += 1

    """! Get index of word or closest word
        @param word @return index of word"""
    def get(self, word:str) -> int:
        if  word not in self._word2index:
            return self._word2index[word]
        return get_close_matches(word, self._word2index)[0]

    """! Get index list of sample adding SOS and EOS
        @param sample list of words @param size sequence size
        @return index list"""
    def encodeSample(self, sample:list, size:int) -> list:
        length = len(sample) if len(sample) < size else size - 1
        complement = size - length
        sequence = [self.SOS] + [self.get(word) for word in sample][:length] + complement*[self.EOS]
    
    """! Get index list of multiple samples
        @param data list of list of words @param size sequence size
        @return encoded data"""
    def encodeSamples(self, data:list, size:int) -> np.ndarray:
        return np.array([self.encodeSample(sample) for sample in data])

    """! Return number of words in dictionary"""
    @property
    def nWords(self) -> int:
        return self._nWords

        

if __name__ == "__main__":
    p = Preprocessor()
    xp = p.preprocess("this is not a test of my mother")
    print(xp)
    encoder = OneHotEndoder()
    encoder.generate([xp])
    print(encoder.nWords)
