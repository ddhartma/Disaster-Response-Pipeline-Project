# Disaster Response Pipeline Project

Let's create a machine learning pipeline that is able to save human life due to natural disasters.

Take a data set containing real messages that were sent during disaster events. Create a machine learning pipeline including an app to categorize these events. An emergency worker can:
  - input a new message,
  - get classification results in several categories and
  - send such classified reports to an appropriate disaster relief agency.

Examples of natural disasters resulting from natural processes of the Earth include floods, hurricanes, tornadoes, volcanic eruptions, earthquakes, tsunamis, storms, and other geologic processes. Those disasters can cause loss of life and lead to private and public economic damages.

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
- ***model/train_classifier.py***: Python script countaing the ML pipelining part for creating a model
- ***model/...pkl***: pickle files containing the trained models. These models are loaded and used in the web app

### Flask app:
The Flask web app contains 

    - dataframe evaluation results and 
    - provides an user interactive message classification terminal.

- ***app/run.py***: Python (Flask) script to start the server process. It provides the html template content. 
- ***app/static/img/...png***: images needed for the web app ***master.html*** file.
- ***app/templates/master.html***: the main html file to create the web app.
- ***app/templates/go.html***: html file for presenting the prediction result from an inserted message.
datasets, ***merges*** the two datasets, ***cleans*** the data, ***stores*** it in a SQLite database)


## The Web App <a name="web_app"></a>


## CRISP-DM Analysis <a name="CRISP_DM"></a>
In the beginning a CRISP-DM analysis (CROSS INDUSTRY STANDARD PROCESS FOR DATA MINING) has been applied containing the process steps:

## Business Understanding <a name="Business_Understanding"></a>

**What does the app need to be checked to improve its performance?** This question is divided in separate  - more accurate - questions:

- Question 1: How are the three different 'genre' types distributed?
- Question 2: What is the distribution of letters-count for each genre? Are there any outliers? 
- Question 3: What is the distribution of words-counts for each genre? Are there any outliers?
- Question 4: Does an outlier removement improve the model?
- Question 5: Are there any significant correlations between the categories?

## DataFrame Understanding <a name="DataFrame_Understanding"></a>
Dataset with 26028 observations and 40 columns

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

- ***NaN values***: Missing values are from the categorical varibale 'original'. This column is not needed for modeling. Hence those NaN values do not need to be imputed.

- ***Dublicate values***: 170 dublicate values were found during ETL processing. Those were removed from the dartaset.

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

    ***Dropping Columns***: The columns ```id```, ```origin``` and ```genre``` are not used for modeling. They are not dropped from the dataframe for answering CRISP-DM analysis questions.

    ***Creating Binaries***: The binary set of variables (see above) was created from the categoties.csv



## Modeling: <a name="Modeling"></a>

The main modeling approach in this Jupyter notebook is done based on a sklearn Linear Regression. The R-squared value (a measure of how much of the data variability can be explained by the model) is ABOUT ... for training and ... for testing.
Are there nonlinear tendencies for some features-target-dependencies? Are there linearization steps included?
Are there deep learning nonlinear model applied for predictions (Tensorflow, Pytorch)?

## Evaluation: <a name="Evaluation"></a>

The answers to this CRISP questions and further information can be found in the jupyter notebook. The most important results are:

    Answer to Question 1: ...
    Answer to Question 2: ...
    Answer to Question 3: ...


Below are a few screenshots of the web app.

 to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

This project will consists of a web app where one can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


Below are a few screenshots of the web app.




After you complete the notebooks for the ETL and machine learning pipeline, you'll need to transfer your work into Python scripts, process_data.py and train_classifier.py. If someone in the future comes with a revised or new dataset of messages, they should be able to easily create a new model just by running your code. These Python scripts should be able to run with additional arguments specifying the files used for the data and model.
Example:

python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

python train_classifier.py ../data/DisasterResponse.db classifier.pkl

Templates for these scripts are provided in the Resources section, as well as the Project Workspace IDE. The code for handling these arguments on the command line is given to you in the templates.
Flask App

In the last step, you'll display your results in a Flask web app. We have provided a workspace for you with starter files. You will need to upload your database file and pkl file with your model.

This is the part of the project that allows for the most creativity. So if you are comfortable with html, css, and javascript, feel free to make the web app as elaborate as you would like.

In the starter files, you will see that the web app already works and displays a visualization. You'll just have to modify the file paths to your database and pickled model file as needed.

There is one other change that you are required to make. We've provided code for a simple data visualization. Your job will be to create two additional data visualizations in your web app based on data you extract from the SQLite database. You can modify and copy the code we provided in the starter files to make the visualizations.



## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


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
$ git clone https://github.com/ddhartma/ETL-Pipelines.git
```

- Change Directory
```
$ cd 5_ETL_DATA_Pipelines
```

- Create a new Python environment, e.g. ds_etl. Inside Git Bash (Terminal) write:
```
$ conda create --name ds_etl
```

- Install the following packages (via pip or conda)
```
numpy = 1.17.4
pandas = 0.24.2
```

- Check the environment installation via
```
$ conda env list
```

- Activate the installed environment via
```
$ conda activate ds_etl
```

## Acknowledgments <a name="Introduction"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.
