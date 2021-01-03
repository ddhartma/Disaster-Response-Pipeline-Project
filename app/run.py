import json
import plotly
import pandas as pd

import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Box, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from CustomVectorizer import CustomVectorizer


app = Flask(__name__)

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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql("SELECT * FROM disaster", engine)

# load model
model = joblib.load("../models/classifier_pipeline_2.pkl")
print('Active model: ' '...models/classifier_pipeline_2.pkl')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """ Get Plot data ready, construct Plotly figures, render_template
        INPUTS:
        ------------

        OUTPUTS:
        ------------
        render_template - render master.html template
    """

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    # create boxplot and Histograms: What is the distribution of word-count for each genre? Are there any outliers?
    messages_stats_direct = pd.read_csv(open('../models/message_stats_direct.csv'))
    messages_stats_news = pd.read_csv(open('../models/message_stats_news.csv'))
    messages_stats_social = pd.read_csv(open('../models/message_stats_social.csv'))
    messages_stats = {'direct': messages_stats_direct, 'news': messages_stats_news, 'social': messages_stats_social}

    median_direct = messages_stats_direct.iloc[:,1].median()
    median_news = messages_stats_news.iloc[:,1].median()
    median_social = messages_stats_social.iloc[:,1].median()

    # Get most common words
    most_comon_words = pd.read_csv('../models/most_common_words.csv', index_col=[0])

    # create correlation plot: Are there any significant correlations between the categories?
    corr_x = df.corr().index
    corr_y = df.corr().index

    # read in som example messages: raw and tokenized
    message_set = pd.read_csv('../models/message_set.csv', index_col=[0])



    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [Box(y=message_length.iloc[:,1], name=message_type) for message_type, message_length in messages_stats.items()],

            'layout': {
                'title': 'Word-count descriptive stats for each genre',

            }
        },
        {
            'data': [Histogram(x=message_length.iloc[:,1], name=message_type) for message_type, message_length in messages_stats.items()],
            'layout': {
                'title': 'Word-count distribution for each genre',
                'yaxis': {
                    'title': "Count",
                    'type': "log"
                },
                'xaxis': {
                    #'type': "log"
                },

            }
        },

        {
            'data': [
                Bar(
                    x=most_comon_words['idf_weights'],
                    y=list(most_comon_words.index)[:20],
                    orientation='h'
                )
            ],

             'layout': {
                'title': 'Most common words after tokenization',
                'yaxis': {
                    'title': "idf_weights"
                },
                'xaxis': {
                    'title': "Most common words"
                }
            }
        },

        {
            'data': [
                Heatmap(
                    x=corr_x,
                    y=corr_y,
                    z=df.corr().values,
                    type = 'heatmap',
                    colorscale = 'Viridis'
                )
            ],

            'layout': {
                'title': 'Correlation Matrix',
                'yaxis': {
                    'automargin': True
                },
                'xaxis': {
                    'automargin': True
                }

            }
        },


    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', median_direct=median_direct,
                                          median_news= median_news,
                                          median_social=median_social,
                                          message_raw=list(message_set.iloc[:,0]),
                                          message_tok=list(message_set.iloc[:,1]),
                                          ids=ids,
                                          graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """ predict classification for query and prepare rendering for go.html

        INPUTS:
        ------------
        no direct inputs

        OUTPUTS:
        ------------
        render_template for go.html
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(classification_labels)
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """ Start the Web App

    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
