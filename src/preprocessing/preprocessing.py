import numpy as np 
import pandas as pd
import os
import sys
import logging
import yaml
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

# configuration of logger
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(train_data_path:pd.DataFrame, test_data_path: pd.DataFrame) -> pd.DataFrame:
    """
    Load data from the data this path 
    """
    try:
        logging.info("Load train data ")
        train_data=pd.read_csv(train_data_path)
        logging.info("Load test data ")
        test_data=pd.read_csv(test_data_path)
        logging.info("Data loaded completed")
        return train_data, test_data
    except FileNotFoundError:
        logging.error("File not found")
        raise

def impute_missing_values(train_data:pd.DataFrame,test_data:pd.DataFrame)->pd.DataFrame:
    """
    impute missing values in the data

    """
    try:
        logging.info('drop the cabin column from the both train and test data')
        X_train=train_data.drop('Cabin',axis=1)
        X_test=test_data.drop('Cabin',axis=1)
        logging.info('Cabin column removed')

        logging.info('filling missing values in Age, Embarked and Fare column')
        X_train['Age'].fillna(X_train['Age'].mean(),inplace=True)
        X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)

        X_train['Embarked'].fillna(X_train['Embarked'].mode()[0],inplace=True)

        X_test['Fare'].fillna(X_test['Fare'].mean(),inplace=True)
        logging.info('missing values filled')
        return X_train, X_test
    except KeyError:
        logging.error("Column not found")
        raise

def split_Name(train_data:pd.DataFrame,test_data:pd.DataFrame)->pd.DataFrame:
    """
    split the Name column into two columns

    """
    try:
        logging.info('splitting the Name column into two columns')
         
        train_data['surname']=train_data['Name'].str.split(',').str.get(0)
        train_data['Name']=train_data['Name'].str.split(',').str.get(1)

        test_data['surname']=test_data['Name'].str.split(',').str.get(0)
        test_data['Name']=test_data['Name'].str.split(',').str.get(1)

        logging.info('surname column add in 2nd position')
        train_data.insert(2,'surname',train_data.pop('surname'))
        test_data.insert(2,'surname',test_data.pop('surname'))
        logging.info('Name column split completed')
        return train_data, test_data

    except KeyError:
        logging.error("Column not found")
        raise


"""
--------------------------------------------------------

"""



def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame, file_path:str) -> None:
    """
    Save data to the file path
    """
    try:
        logging.info('saving train data')
        file_path=os.path.join('data','interim')
        if not os.path.exists(file_path):
            logging.info('Creating directory')
            os.makedirs(file_path)
            logging.info('Directory created')

        logging.info('saving train data')
        train_data.to_csv(os.path.join(file_path,'train_data.csv'),index=False)
        logging.info('saving test data')
        test_data.to_csv(os.path.join(file_path,'test_data.csv'),index=False)
        logging.info('Data saved successfully')
    except FileNotFoundError:
        logging.error("File not found")
        raise

def main():
    """
    Run the main function.
    """
    # Load parameters from params.yaml
    with open("config/params.yaml", "r") as file:
        params = yaml.safe_load(file)

    train_path = params["data"]["train_path"]
    test_path = params["data"]["test_path"]
    output_path = params["data"]["output_path"]

    train_data, test_data = load_data(train_path, test_path)
    train_data, test_data = impute_missing_values(train_data, test_data)
    train_data, test_data = split_Name(train_data, test_data)
    save_data(train_data, test_data, output_path)


if __name__ == "__main__":
    main()
