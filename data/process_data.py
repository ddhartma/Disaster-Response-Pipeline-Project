# import libraries
import sys
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
    
def load_data(messages_filepath, categories_filepath):
    """ Load CSV files and combine datasets
        INPUTS:
        ------------
        messages_filepath - filepath to messages csv file
        categories_filepath - filepath to categories csv file
            
        OUTPUTS:
        ------------
        df - dataframe, merged from messages and categories dataframe 
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    print('messages dataframe')
    print(messages.head())
    print('Shape of messages: ' + str(messages.shape)) 
    print(' ')
    
    
    # load categories dataset
    categories =pd.read_csv(categories_filepath)
    print('categories dataframe')
    print(categories.head())
    print('Shape of categories: ' + str(categories.shape))
    print(' ')
    
    # merge datasets
    df = messages.merge(categories, how='outer', on=['id'])
    print('merged df')
    print(df.head())
    print('Shape of merged df: ' + str(df.shape))
    print(' ')
    
    return df


def clean_data(df):
    """ Rearrange and clean dataframe  
        - create a dataframe of the 36 individual category columns
        - convert category values to just numbers 0 or 1
        - drop the original categories column from `df`
        - concatenate the original dataframe with the new `categories` dataframe
        - drop duplicates
        - ignore rows with related = 2
        
        INPUTS:
        ------------
        df - the merged dataframe from load_data function
        
        OUTPUTS:
        ------------
        df - cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True) # split column values
    print('Create a dataframe of the 36 individual category columns')
    print('Shape of categories: ' + str(categories.shape))
    print(' ')
    
    # select the first row of the categories dataframe
    row = categories.loc[0, :]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.tolist() # get first row of categories as list
    category_colnames = [col.split('-')[0] for col in category_colnames] #  remove number from each element of this list
    print('These are the column names of categories')
    print(category_colnames)
    print(' ')
    
    # rename the columns of `categories`
    categories.columns = category_colnames # set new colnames
    print('Rename the columns of `categories`')
    print('Shape of categories: ' + str(categories.shape))
    print(' ')
          
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].apply(lambda x : x.split('-')[1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

     
    print('Convert category values to just numbers 0 or 1')
    print('Shape of categories: ' + str(categories.shape))
    print(' ')
          
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    print('Drop the original categories column from `df`')
    print('Shape of df: ' + str(df.shape))
    print(' ')
          
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    print('concatenate the original dataframe with the new `categories` dataframe')
    print('Shape of df: ' + str(df.shape))  
    print(' ')
          
    # check number of duplicates
    print('Check number of duplicates')
    print('Number of dublicates: ', len(df)-len(df.drop_duplicates()))  
          
    # drop duplicates
    df = df.drop_duplicates()
    print('drop duplicates')
    print('Shape of df: ' + str(df.shape))  
        
    # check number of duplicates
    print('Check number of duplicates')
    print('Number of dublicates: ', len(df)-len(df.drop_duplicates()))  
    print(' ')
    
    # ignore rows with related = 2
    df = df[df['related'] != 2]
    print('Ignore rows with related = 2')
    print('Shape of df: ' + str(df.shape))
    print('')
          
          
    print('-----------------------------------------')
    print('Result after clean_data')
    print(df.head())
    print('Shape of df: ' + str(df.shape))
    print('-----------------------------------------')
    print(' ')   
    
    return df
    

def save_data(df, database_filename):
    """ Store database and dataframe description to disk 

        INPUTS:
        ------------
        df - cleaned dataframe ready to store in Sqlite database
        database_filename - filename of the database
        
        OUTPUTS:
        ------------
        outputs will be saved to disk: 
        df_describe.xlsx - a description of the dataframe 
        
        
    """
    # export descriptive statistics table
    df_describe = df.describe(include='all').T
    df_describe.to_excel('df_describe.xlsx')  
       
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster', engine, index=False, if_exists='replace')

    # Check storage
    df_check = pd.read_sql("SELECT * FROM disaster", engine)
    print('df_check')
    print(df_check.head())
    print('Shape of df_check: ' + str(df_check.shape))


def main():
    """ Main Function to trigger/control all other functions: 
        - load_data 
        - clean_data
        - save_data
        
        INPUTS:
        ------------
        no direct inputs
        via sys.argv get filepaths from command line
        
        OUTPUTS:
        ------------
        no direct outputs
        
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()