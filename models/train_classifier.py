# import libraries
import sys
import warnings
warnings.filterwarnings("ignore")
from sqlalchemy import create_engine

import os
from time import time
import copy

import numpy as np
import pandas as pd

import pickle

import re

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords', 'omw-1.4'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
#from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from pipelinehelper import PipelineHelper

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """ To know the parts of speech (like nouns, verbs, pronouns)
        can help to understand the meaning of a sentence better.
        This class checks if the first word of a sentence is a verb.
        if yes --> return True
        if no --> return False

    """
    def starting_verb(self, text):
        """ function that
            - divides a text string into a list of sentences
            - checks if the first word of a sentence is a verb

            INPUTS:
            ------------
            text - a string of text

            OUTPUTS:
            ------------
            True - if verb
            False - if anything else than verb
        """

        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        """ fit function for estimator (object that learns from data),
            here estimator is an instance of StartingVerbExtractor class

            INPUTS:
            ------------
            x - 2d array X of the dataset features
            y - 1d array y of the dataset target labels

            OUTPUTS
            ------------
            self - allows to chain methods together. This method is required to be compatible with scikit-learn
        """
        return self

    def transform(self, X):
        """ function which includes the code to transform the data

            INPUTS:
            ------------
            x - 2d array X of the dataset features

            OUTPUTS:
            ------------
            df_x_tagged - a DataFrame of X_tagged (containing a column with True and False values)
                          this transformer object will be appendend to the pipeline object via Feature Union
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        df_x_tagged = pd.DataFrame(X_tagged)
        return df_x_tagged


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

# load data from database
def load_data(database_filepath):
    """ Load data from Sqlite database into DataFrame
        INPUTS:
        ------------
        database_filepath - path to Sqlite database

        OUTPUTS:
        ------------
        X - input features (messages) of DataFrame df
        Y - categories of DataFrame df
        category_names - list of category names (column names of Y)

    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql("SELECT * FROM disaster", engine)

    # X and y
    colnames = df.columns.tolist()
    category_names = colnames[4:]

    X = df.message.values
    Y = df[category_names]

    # Show DataFrame result, shapes, colnames, target
    print('DATAFRAME df')
    print(df.head())
    print('Shape of df: ' + str(df.shape))
    print(' ')

    print('DATAFRAME Y')
    print(Y.head())
    print('Shape of Y: ' + str(Y.shape))
    print(' ')

    print('colnames')
    print(colnames)
    print(' ')

    print('category_names')
    print(category_names)
    print(' ')

    return X, Y, category_names, df


def tokenize(text, word_prep='lemmatize'):
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
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Stem, normalize all words to lower case, remove white spaces
    if word_prep == 'stem':
        clean_tokens = [PorterStemmer().stem(tok).lower().strip() for tok in tokens]

    # Lemmatize, normalize all words to lower case, remove white spaces
    if word_prep == 'lemmatize':
        clean_tokens = [WordNetLemmatizer().lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens


def tfid_transform(X):
    """ Get weighted idfs for each word of the Bag-of-Words (CountVectorizer matrix)

        INPUTS:
        ------------
        X - numpy.ndarray of training (testing) features

        OUTPUTS:
        ------------
        df_idf.sort_values(by=['idf_weights']) - DataFrame with all words from Bag-of-Words as index
                                                 and idf_weights as one column
    """
    # Build the pipeline
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])

    print('Compute the IDF values ...')
    pipeline.fit(X)

    # create df_idf DataFrame
    df_idf = pd.DataFrame(pipeline.named_steps['tfidf'].idf_, index=pipeline.named_steps['vect'].get_feature_names(),columns=["idf_weights"])

    # sort ascending
    return df_idf.sort_values(by=['idf_weights'])


def build_model(X_train, Y_train, pipeline_name='pipeline_1'):
    """ Build a ML pipelines
        Test different pipelines
        - pipeline_1: standard based on CountVectorizer, TfidfTransformer and MultiOutputClassifier
        - pipeline_2: as pipeline_1 but with a CustomVectorizer and GridSearchCV to find optimized parameters
        - pipeline_3: add the CustomTransformer 'StartingVerbExtractor' into pipeline via Feature Union

        INPUTS:
        ------------
        pipeline_name - string name for calling a specific pipeline
        X_train - numpy.ndarray, input features for training
        Y_train - numpy.ndarray, target values

        OUTPUTS:
        ------------
        cv  - model based on sklearn GridSearchCV and actual parameter settings
        pipeline - model based on sklearn Pipeline (including ETL and NLP processing steps)
        parameters - dictionary of GridSearchCV parameters

    """
    if pipeline_name == 'pipeline_1':
        print('pipeline_1 chosen')
        pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier())),
                #('clf', SGDClassifier()),
            ])
        return pipeline, None, None


    if pipeline_name == 'pipeline_2':
        print('pipeline_2 chosen')
        pipeline = Pipeline([
            ('nlp', Pipeline([
                #('tokenizer', CustomTokenizer(word_prep='stem')),
                ('vect', CustomVectorizer(X_train, word_prep='lemmatize')),
                ('tfidf', TfidfTransformer()),
            ])),
            ('classifier', PipelineHelper([
                ('rfc', MultiOutputClassifier(RandomForestClassifier())),
                ('abc', MultiOutputClassifier(
                        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))))
            ]))
        ])

        # uncommenting more parameters will give better exploring power but will
        # increase processing time in a combinatorial way
        parameters = {
                #'nlp__vect__word_prep': ('stem', 'lemmatize'),
                'nlp__vect__remove_stopwords': (True, 'False'),
                'nlp__vect__max_df': (0.5, 0.75, 1.0),
                #'nlp__vect__max_features': (None, 5000, 10000, 50000),
                #'nlp__vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                #'nlp__tfidf__use_idf': (True, False),
                #'nlp__tfidf__norm': ('l1', 'l2'),

            'classifier__selected_model': pipeline.named_steps['classifier'].generate({
                'rfc__estimator__n_estimators': [10, 20],
                'rfc__estimator__min_samples_split': [2, 5],
                'abc__estimator__learning_rate': [0.1, 0.3],
                'abc__estimator__n_estimators': [100, 200],
            })
        }
        cv = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=1, verbose=1)
        #cv.fit(X_train, Y_train)
        return cv, pipeline, parameters


    if pipeline_name == 'pipeline_3':
        print('pipeline_3 chosen')
        pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
        parameters = {
            #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
            #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
            'features__text_pipeline__vect__max_features': (None, 5000, 10000),
            'features__text_pipeline__tfidf__use_idf': (True, False),
            #'clf__estimator__n_estimators': [50, 100, 200],
            #'clf__estimator__min_samples_split': [2, 3, 4],
            'features__transformer_weights': (
                {'text_pipeline': 1, 'starting_verb': 0.5},
                {'text_pipeline': 0.5, 'starting_verb': 1},
                #{'text_pipeline': 0.8, 'starting_verb': 1},
            )
        }

        cv = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=1, verbose=1)
        #cv.fit(X_train, Y_train)
        return cv, pipeline, parameters


def evaluate_model(model, df, X_train, X_test, Y_test, category_names, pipeline_name):
    """ Get a classification report from testing results including precision, recall, f1-score an accuracy

        INPUTS:
        ------------
        model - the trained model based on the actual pipeline
        X_test - DataFrame with test input features
        Y_test - DataFrame of true target values

        OUTPUTS:
        ------------
        print statements for precision, recall, f1-score an accuracy for each category
        print statement for size of Bag-of-Words
        print statements and csv export for message stats
        print statements and csv export for 20 most common words
        print statements and csv export for 100 randomly chosen messages (in raw format and tokenized)

    """
    # Make predictions based on trained model
    Y_pred = model.predict(X_test)

    #print(classification_report(y_test, y_pred, target_names=y_test.keys()))
    accuracy = (Y_pred == Y_test).mean()
    df_classification_report = pd.DataFrame(classification_report(Y_test, Y_pred, target_names=Y_test.keys(),  output_dict=True))
    df_classification_report = pd.concat([df_classification_report.T, accuracy], axis=1).reindex(df_classification_report.T.index)
    df_classification_report.columns = ['f1_score', 'precision', 'recall', 'support', 'accuracy']
    print(pipeline_name)
    print(df_classification_report)
    print(' ')
    print('Total accuracy = ' + str(round(accuracy.mean(),2)))
    print(' ')

    # get most commomn words
    most_comon_words = tfid_transform(X_train)
    most_comon_words.to_csv('models/most_common_words.csv')
    print('20 most common words')
    print(list(most_comon_words.index)[:20])
    print(' ')
    print('Size of Bag-of-Words: ', len(list(most_comon_words.index)))
    print(' ')
    print('... most_common_words.csv saved!')
    print(' ')

    # Check 100 randomly chosen messages --- coming from X_train with and without tokenization
    rand_set = np.random.randint(df.shape[0], size=(1, 100))
    message_raw = []
    message_tok = []
    for index in rand_set[0]:
        try:
            print(X_train[index])
            print('')
            print(tokenize(X_train[index]))
            print('--------------------------------------------------------')
            message_raw.append(X_train[index])
            message_tok.append(tokenize(X_train[index]))
        except:
            pass

    message_set = pd.DataFrame({'message_raw': message_raw, 'message_tok': message_tok})
    message_set.to_csv('models/message_set.csv')
    print('... message_set.csv saved!')
    print(' ')

    # distribution of word counts for each genre
    # create boxplot and Histograms: What is the distribution of word-count for each genre? Are there any outliers?
    print('Tokenize messages ...')
    message_stats_direct = df[df['genre'] == 'direct']['message'].apply(lambda x: len(tokenize(x)))
    message_stats_news = df[df['genre'] == 'news']['message'].apply(lambda x: len(tokenize(x)))
    message_stats_social = df[df['genre'] == 'social']['message'].apply(lambda x: len(tokenize(x)))
    print('Median of direct message word count: ', message_stats_direct.median())
    print('Median of news message word count: ', message_stats_news.median())
    print('Median of social message word count: ', message_stats_social.median())
    print(' ')
    message_stats_direct.to_csv('models/message_stats_direct.csv')
    print('... message_stats_direct.csv saved!')
    message_stats_news.to_csv('models/message_stats_news.csv')
    print('... message_stats_news.csv saved!')
    message_stats_social.to_csv('models/message_stats_social.csv')
    print('... message_stats_social.csv saved!')

    print('... messages_stats.json saved!')
    print(' ')


def save_model(model, model_filepath):
    """ Save the model

        INPUTS:
        ------------
        model: model to be saved
        model_filepath: filepath to model

        OUTPUTS:
        ------------
        save model as a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Main function to trigger model training, model evaluation and model saving
        Function that tiggers
        - load_data
        - train_test_split
        - build_model
        - model.fit
        - evaluate_model
        - save_model

        INPUTS:
        ------------
        No direct inputs, however there are three pipelines to bes tested in build_model.
        These pipelines are called via the pipeline_names list

        OUTPUTS:
        ------------
        no direct outputs, however the model is stored as a pickle file to disk

    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        # load data
        X, Y, category_names, df = load_data(database_filepath)

        # train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # start pipelining, build the model
        pipeline_names = ['pipeline_2', 'pipeline_3']
        for pipeline_name in pipeline_names:
            print('Building model...')
            model, pipeline, parameters = build_model(X_train, Y_train, pipeline_name)

            if pipeline_name in ['pipeline_2', 'pipeline_3']:
                print("Performing grid search...")
                print("pipeline:", [name for name, _ in pipeline.steps])
                print("parameters:")
                print(parameters)

                t0 = time()

            # train the model
            print('Training model...')
            model.fit(X_train, Y_train)
            #print(pipeline['vect'].get_feature_names())

            if pipeline_name in ['pipeline_2', 'pipeline_3']:
                print("done in %0.3fs" % (time() - t0))
                print()

                print("Best score: %0.3f" % model.best_score_)
                print("Best parameters set:")
                best_parameters = model.best_estimator_.get_params()
                for param_name in sorted(parameters.keys()):
                    print("\t%s: %r" % (param_name, best_parameters[param_name]))

            print('Evaluating model...')

            # evaluate the model
            evaluate_model(model, df, X_train, X_test, Y_test, category_names, pipeline_name)

            # save the model
            path, filename = os.path.split(model_filepath)
            base, ext  = os.path.splitext(filename)
            model_filepath = os.path.join(path, base + '_' + pipeline_name + '.pkl')
            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
