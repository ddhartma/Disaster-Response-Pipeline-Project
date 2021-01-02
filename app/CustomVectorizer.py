# defines a custom vectorizer class
import json
import pandas as pd

import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# defines a custom vectorizer class
class CustomVectorizer(CountVectorizer):
    """ A CustomVectorizer class which inherits from the CountVectorizer class.
        Aim: switch between Porterstemmer and Lemmatization during training via GridSearchCV.
        A CountVectorizer object converts a collection of text documents to a matrix of token counts.
    """
    def __init__(self, X, word_prep='lemmatize', remove_stopwords=True, **kwargs):
        """ Init function that takes all arguments of CountVectorizer base class and adds two own arguments

            INPUTS:
            ------------
            X - numpy.ndarray of training (testing) features
            word_prep - string ('stem' or 'lemmatize'),
                        to choose between stemming or lemmatization during tokenization
                        Useful for GridSearchCV

            OUTPUTS:
            ------------
            no direct outputs
        """
        super().__init__(**kwargs)

        self.X = X
        self.word_prep = word_prep
        self.remove_stopwords = remove_stopwords
        self.lowercase=False


    def prepare_doc(self, text):
        #print(self.word_prep)
        """ function that will
            - replace urls with spaceholder
            - remove punctuation
            - remove stopwords
            - stem/lemmatize words
            - normalize all words to lower case
            - remove white spaces

            INPUTS:
            ------------
            text - text as string

            OUTPUTS:
            ------------
            clean_tokens - a list of cleaned words

        """

        # Detect URLs
        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        detected_urls = re.findall(url_regex, text)
        for url in detected_urls:
            text = text.replace(url, "urlplaceholder")

        # Remove punctuation
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)

        tokens = word_tokenize(text)

        # Remove stopwords
        if self.remove_stopwords == True:
            tokens = [t for t in tokens if t not in stopwords.words('english')]
        else:
            pass

        # Stem, normalize all words to lower case, remove white spaces
        if self.word_prep == 'stem':
            clean_tokens = [PorterStemmer().stem(tok).lower().strip() for tok in tokens]

        # Lemmatize, normalize all words to lower case, remove white spaces
        if self.word_prep == 'lemmatize':
            clean_tokens = [WordNetLemmatizer().lemmatize(tok).lower().strip() for tok in tokens]
        #print(clean_tokens)
        return clean_tokens

    def get_params(self, deep=True):
        """ overwrite get_params in CountVectorizer base class
            create new get_params() including the new property word_prep

            INPUTS:
            ------------
            deep - parameter in get_params function of base class

            OUTPUTS:
            ------------
            params - new parameter dictionary
        """
        params = super().get_params(deep)
        # Hack to make get_params return base class params...
        cp = copy.copy(self)
        cp.__class__ = CountVectorizer
        params.update(CountVectorizer.get_params(cp, deep))
        return params

    def build_analyzer(self):
        """ overwrite build_analyzer in CountVectorizer base class

            INPUTS:
            ------------

            OUTPUTS:
            ------------
            - call prepare_doc, transform training (testing) features,
              return cleaned lists of word tokenized messages
        """
        preprocess = self.build_preprocessor()
        return lambda doc : preprocess(self.decode(self.prepare_doc(doc)))
