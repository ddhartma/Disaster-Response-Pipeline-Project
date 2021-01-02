[image1]: assets/word_count_box.png "image1"
[image2]: assets/word_count_hist.png "image2"
[image3]: assets/common_words.png "image3"
[image4]: assets/genre_distribution.png "image4"
[image5]: assets/correlation.png "image5"
[image6]: assets/web_1.png "image6"
[image7]: assets/web_2.png "image7"
[image8]: assets/web_3.png "image8"
# Disaster Response Pipeline Project

Let's create a machine learning pipeline that is able to save human life during natural disasters.

To achieve this take a data set containing real messages that were sent during disaster events. Train a machine learning pipeline and create a web app to categorize new messages. An emergency worker can then:
  - input a new message,
  - get classification results in several categories and
  - send such classified reports to an appropriate disaster relief agency.

Examples of natural disasters resulting from natural processes of the Earth include floods, hurricanes, tornadoes, volcanic eruptions, earthquakes, tsunamis, storms, and other geologic processes. Those disasters can cause loss of life and lead to private and public economic damages.

An automatic evaluation of text based news could simplify a decision process for emergency organisations and accelerate deliveries of relief goods.

In this project the used dataset is based on ***real*** disaster data from [Figure Eight]().
## Outline
-  [The Pipelining Process (files in the repo)](#Pipelining_Process)
-  [The Web App](#web_app)
-  [CRISP-DM Analysis](#CRISP_DM)
    - [Business Understanding](#Business_Understanding)
    - [DataFrame Understanding](#DataFrame_Understanding)
    - [DataFrame Preparation](#DataFrame_Preparation)
    - [Modeling](#Modeling)
    - [Evaluation](#Evaluation)
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

## The Pipelining Process (files in the repo) <a name="Pipelining_Process"></a>
### ETL pipeline:

ETL processes were first developed in a jupyter notebook (ETL/ETL Pipeline Preparation.ipynb) and then transeferred into a Python script (process_data.py). The ETL pipeline contains the steps:

    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database

- ***ETL/messages.csv***: csv file containing one part of the dataset (translated english messages, original messages and genre of message)
- ***ETL/categories.csv***: csv file containing the second part of the dataset (categories, i.e. the target values)
- ***ETL/ETL Pipeline Preparation.ipynb***: Jupyter notebook containg all ETL steps (***loads*** the messages and categories datasets, ***merges*** the two datasets, ***cleans*** the data, ***stores*** it in a SQLite database)
- ***data/DisasterResponse.db***: database for ML pieline- SQlite output of the ETL process
- ***data/disaster_categories.csv***: csv file containing one part of the dataset (messages, original, genre)
- ***data/disaster_messages.csv***: sv file containing the second part of the dataset (categories)
- ***data/process_data.py***: Python script containg all ETL steps (***loads*** the messages and categories 

### ML pipeline:
The ML processes were first developed in a jupyter notebook (ML/ML Pipeline Preparation.ipynb) and then transferred to the Python script (train_classifier.py) The learning ML pipeline contains the steps:

    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file

- ***ML/ML Pipeline Preparation.ipynb***: Jupyter notebook containg all ML steps (loads data from the SQLite database, ***splits*** the dataset into training and test sets, ***builds*** a text processing and machine learning pipeline, ***trains*** and ***tunes*** a model using GridSearchCV, outputs results on the ***test*** set, ***exports*** the final model as a pickle file
- ***ML/models/...pkl:*** pickle files containing the trained models created in the ML/ML Pipeline Preparation.ipynb notebook
- ***models/train_classifier.py***: Python script countaing the ML pipelining part for creating a model
- ***models/...pkl***: pickle files containing the trained models. These models are loaded and used in the web app
- ***models/message_set.csv***: Example set of raw and tokenized messages 
- ***models/message_stats_direct.csv***: word count of tokenized direct messages
- ***models/message_stats_news.csv***:  word count of tokenized news messages
- ***models/message_stats_social.csv***:  word count of tokenized social messages
- ***models/most_common_words.csv***: csv file with ascending idf values for all words in the vocabulary.

### Flask app:
The Flask web app  

    - shows evaluation results of the dataset 
    - enables a user interactive message classification

- ***app/run.py***: Python (Flask) script to start the server process. It provides the html template content. 
- ***app/static/img/...png***: images needed for the web app ***master.html*** file.
- ***app/templates/master.html***: the main html file to create the web app.
- ***app/templates/go.html***: html file for presenting the prediction result for a given message.

## The Web App <a name="web_app"></a>
Below are a few screenshots of the web app. The web app header:
    ![image6]
    The message and result block:
    ![image7]
    The interactive dataset analysis part:
    ![image8]



## CRISP-DM Analysis <a name="CRISP_DM"></a>
In the beginning a CRISP-DM analysis (CROSS INDUSTRY STANDARD PROCESS FOR DATA MINING) has been applied containing the process steps:
- Business Understanding
- DataFrame Understanding
- Data preparation
- Modeling
- Evaluation

## Business Understanding <a name="Business_Understanding"></a>

**What does the app still need to be to improve its performance?** This question is divided in separate  - more accurate - questions:

- Question 1: How are the three different 'genre' types distributed?
- Question 2: What is the distribution of word-counts for each genre? Are there any outliers?
- Question 3: What are the 20 most common words in the training set?
- Question 4: Are there any significant correlations between the categories?

Answers to these questions are provided in the notebook ```ML/ML Pipeline Preparation.ipynb``` and in the web app.

## DataFrame Understanding <a name="DataFrame_Understanding"></a>
Dataset with 26028 observations (messages) and 40 columns

- **Categorical** columns:

	| column_name | type | min | max | number NaN |
	| :-------------  | :-------------  | :-------------  | :-------------  | :-------------  |
	| id | int64 | 2 | 30265 | 0 | 
	| message | object |      | | News Update | Serious loss of life expected in devastating earthquake in Haiti http ow.ly 16klRU | 0 | 
	| original | object | NaN | NaN | 15990 | 
	| genre | object | direct | social | 0 | 


- **Binaries** columns:

	| column_name | type | min | max | number NaN |
	| :-------------  | :-------------  | :-------------  | :-------------  | :-------------  |
	| related | int64 | 0 | 1 | 0 | 
	| request | int64 | 0 | 1 | 0 | 
	| offer | int64 | 0 | 1 | 0 | 
	| aid_related | int64 | 0 | 1 | 0 | 
	| medical_help | int64 | 0 | 1 | 0 | 
	| medical_products | int64 | 0 | 1 | 0 | 
	| search_and_rescue | int64 | 0 | 1 | 0 | 
	| security | int64 | 0 | 1 | 0 | 
	| military | int64 | 0 | 1 | 0 | 
	| child_alone | int64 | 0 | 0 | 0 | 
	| water | int64 | 0 | 1 | 0 | 
	| food | int64 | 0 | 1 | 0 | 
	| shelter | int64 | 0 | 1 | 0 | 
	| clothing | int64 | 0 | 1 | 0 | 
	| money | int64 | 0 | 1 | 0 | 
	| missing_people | int64 | 0 | 1 | 0 | 
	| refugees | int64 | 0 | 1 | 0 | 
	| death | int64 | 0 | 1 | 0 | 
	| other_aid | int64 | 0 | 1 | 0 | 
	| infrastructure_related | int64 | 0 | 1 | 0 | 
	| transport | int64 | 0 | 1 | 0 | 
	| buildings | int64 | 0 | 1 | 0 | 
	| electricity | int64 | 0 | 1 | 0 | 
	| tools | int64 | 0 | 1 | 0 | 
	| hospitals | int64 | 0 | 1 | 0 | 
	| shops | int64 | 0 | 1 | 0 | 
	| aid_centers | int64 | 0 | 1 | 0 | 
	| other_infrastructure | int64 | 0 | 1 | 0 | 
	| weather_related | int64 | 0 | 1 | 0 | 
	| floods | int64 | 0 | 1 | 0 | 
	| storm | int64 | 0 | 1 | 0 | 
	| fire | int64 | 0 | 1 | 0 | 
	| earthquake | int64 | 0 | 1 | 0 | 
	| cold | int64 | 0 | 1 | 0 | 
	| other_weather | int64 | 0 | 1 | 0 | 
	| direct_report | int64 | 0 | 1 | 0 | 
    
- There are ***0 numerical*** (0x int and 0x float) columns
- There are ***4 categorical*** columns
- There are ***36 binary*** columns
- There are ***15990 missing values*** in total in the dataset. 


## Data preparation <a name="DataFrame_Preparation"></a>

The notebook ***ETL Pipeline Preparation.ipynb*** contains the data engineering steps and and all the results.

- ***NaN values***: All missing values are attributed tto the categorical varibale 'original'. This column is not needed for modeling. Hence those NaN values do not need to be imputed.
- ***Dublicate values***: 170 dublicate values were found during ETL processing. Those were removed from the dataset.
- ***special values***: Rows, where column 'related' is classified with 2 were dropped. Those messages are messages which are, e.g.
    - not translated
    - without a meaning or not not understandable code like     
        - "Damocles!Hracles!Philockles!Hyphocles.!yayecles!zigzacles!domagecles!lucles!77h"
        - "9GeQYeYGQEQtm"
        - "Aa.Bb.Cc.Dd.Ee.Ff.Gg.Hh.Ii.Jj.Kk.Ll.Mm.Nn.Oo.Pp.Qq.Rr.Ss.Tt.Uu.Vv.Ww.Xx.Yy.Zz.KERLANGE."
    - incomplete (broken) messages like
        - "NOTES: this message is not complete"
        - "The internet caf Net@le that's by the Dal road by the Maranata church ( incomplete )"
        - "It's Over in Gressier. The population in the area - Incomplete"
    As the amount of 188 messages with relates=2 is too low to justify a time consuming transformation process (like language translation from different langauages to English), these rows will be ignored   
- ***Dropping Columns***: The columns ```id```, ```origin``` and ```genre``` are not used for modeling. ```genre```, however, was used to answer CRISP-DM analysis questions.
- ***Creating Binaries***: The binary set of variables (see above) was created from the categoties.csv
- ***Word count***: There is a strong dependeny to outliers (long messages) in each genre. 
- ***Typical messages used for model training***:
    - I AM IN MISERY SEND SME MONEY FOR ME 
    - The people in Henfrasa across from the La Paix Hospital need food and water. These people are thirsty and hunger is killing them. Please find a way to see what you can do for them. Urgent.
    - Their hands are empty, they lost their families, their homes are rubble, we are asking for aid for these poor people who are dying of hunger, we are throwing out a shout to all the agencies that want to help those victims

- ***Same messages after tokenization***:
    - ['i', 'am', 'in', 'misery', 'send', 'sme', 'money', 'for', 'me']
    - ['the', 'people', 'henfrasa', 'across', 'la', 'paix', 'hospital', 'need', 'food', 'water', 'these', 'people', 'thirsty', 'hunger', 'killing', 'please', 'find', 'way', 'see', 'urgent']
    - ['their', 'hand', 'empty', 'lost', 'family', 'home', 'rubble', 'asking', 'aid', 'poor', 'people', 'dying', 'hunger', 'throwing', 'shout', 'agency', 'want', 'help', 'victim']

- The ***tokenization process*** includes:
    - replace urls with spaceholder
    - remove punctuation
    - remove stopwords
    - stem/lemmatize words 
    - normalize all words to lower case 
    - remove white spaces


## Modeling: <a name="Modeling"></a>
The model consists of sklearn [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) including a [GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html) tuning.


- A CustomVectorizer class which inherits from the [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) class has been developed in order to switch between Porterstemmer and Lemmatization and between stopwords on/off during training via GridSearchCV. Thereby,  a CountVectorizer object converts a collection of text documents to a matrix of token counts.
- In addition, a [TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) has been implemented. The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.

- As there are 36 target categories which have to be classified by the model, a [MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) was added to the pipeline. This multi target classification strategy consists of fitting one classifier per target. This is a simple strategy for extending classifiers like (RandomForestClassifier) that do not natively support multi-target classification.

- A [Pipelinehelper](https://github.com/bmurauer/pipelinehelper) module has been imported to select between different classifier via GridSearchCV. This helper selects between two or more different transformers without pipelining them together.

- A sklearn [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) is used as a target predictor: A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 

- Furthermore, a [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) has been tested. Based on sklearns description An AdaBoost  classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

The full pipeline consists of the following transformers/predictors:
```
pipeline = Pipeline([
            ('nlp', Pipeline([
                ('vect', CustomVectorizer(X_train, word_prep='lemmatize')),
                ('tfidf', TfidfTransformer()),
            ])),
            ('classifier', PipelineHelper([
                ('rfc', MultiOutputClassifier(RandomForestClassifier())),
                ('abc', MultiOutputClassifier(
                        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))))
            ]))
        ])
```

- ***Best GridSearchCV Hyperparameter set***:

    - Best score: 0.240
    - Best parameters set:
        - vect__max_df: 0.5
        - vect__ngram_range: (1, 1)
        - vect__word_prep: 'stem'
            

## Evaluation: <a name="Evaluation"></a>

The answers to the CRISP questions (see above) and further information can be found in the jupyter notebook. The most important results are:

- ***Answer to question 1***: How are the three different 'genre' types distributed?
    ![image4]

    Almost half of the messages (13036) messages are 'news' messages. There are 10634 'direct' and 
    2358 'social' messages.

- ***Answer to question 2***: What is the distribution of word-counts for each genre? Are there any outliers?

    A word count of the messages gave the following distributions: 

    ![image1]

    ![image2]

    There is some outlier spreading in each genre due to long messages. News have longer text sequences than direct or social messages. To further improve the model one could use data padding/truncating techniques so that all messages have the same length.

- ***Answer to question 3***: What are the 20 most common words in the training set?

    Most common 20 words after tokenization: 
    
    'people', 'food', 'water', 'help', 'need', 'please', 'earthquake', 'area', 'u', 'like', 'would', 'said', 'country', 'also', 'know', 'government', 'information', 'one', '000', 'year'

    ![image3]

    Notice, words with lower IDF appear more often than those with higher IDF values. For idf_weights=1 they would appear in each and every document in the collection. The lower the IDF value of a word, the less unique it is to any particular document.

- ***Answer to question 4***: Are there any significant correlations between the categories?

    ![image5]

    Strong correlation (z: 0.8) are found, e.g. for 'transportation' and and 'other_infrasctructure'. However, for the majority of categories intercategorical dependencies are weak.


## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit
- If you need a good Command Line Interface (CLI) under Windowsa you could use [git](https://git-scm.com/). Under Mac OS use the pre-installed Terminal.

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Disaster-Response-Pipeline-Project.git
```

- Change Directory
```
$ cd path-to-main-folder
```

- Create a new Python environment, e.g. ds_ndp. Inside Git Bash (Terminal) write:
```
$ conda create --name ds_ndp
```

- Install the following packages (via pip or conda)
```
numpy = 1.17.4
pandas = 0.24.2
scikit-learn = 0.20
pipelinehelper = 0.7.8
```
Example via pip:
```
pip install numpy
pip install pandas
pip install scikit-learn==0.20
pip install pipelinehelper
```
scikit-learn==0.20 is needed for sklearns dictionary output (output_dict=True) for the classification_report. Earlier versions do not support this.


Link1 to [pipelinehelper](https://github.com/bmurauer/pipelinehelper) 

Link2 to [pipelinehelper](https://stackoverflow.com/questions/23045318/scikit-grid-search-over-multiple-classifiers)

- Check the environment installation via
```
$ conda env list
```

- Activate the installed environment via
```
$ conda activate ds_ndp
```

### Run the web App 

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>
* [Working With Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
* [Natural Language Precessing Book](http://www.nltk.org/book/)
* [sklearn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
* [sklearn CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [sklearn TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
* [10+ Examples for Using CountVectorizer](https://kavita-ganesan.com/how-to-use-countvectorizer/#.X--j6OAxmFo)
* [TF IDF | TFIDF Python Example](https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76)
* [Text Feature Extraction With Scikit-Learn Pipeline](https://towardsdatascience.com/the-triune-pipeline-for-three-major-transformers-in-nlp-18c14e20530)
* [How to Use Tfidftransformer & Tfidfvectorizer?](https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.X-9OH-AxmFp)
* [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
* [Hyperparameter tuning in pipelines with GridSearchCV](https://ryan-cranfill.github.io/sentiment-pipeline-sklearn-5/)
* [Hacking Scikit-Learn’s Vectorizers](https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af)
* [How to inherit from CountVectorizer I](https://stackoverflow.com/questions/51430484/how-to-subclass-a-vectorizer-in-scikit-learn-without-repeating-all-parameters-in)
* [How to inherit from CountVectorizer II](https://sirinnes.wordpress.com/2015/01/22/custom-vectorizer-for-scikit-learn/)
* [pipelinehelper I](https://github.com/bmurauer/pipelinehelper) 
* [pipelinehelper II](https://stackoverflow.com/questions/23045318/scikit-grid-search-over-multiple-classifiers)
