import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk import download

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


if __name__ == "__main__":
    p = Preprocessor()
    print(p.preprocess("this is not a test of my mother"))
