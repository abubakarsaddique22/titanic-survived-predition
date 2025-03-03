import numpy as np 
import pandas as pd 
import os
import sys 
import logging
import yaml

# configuration of logger
# logging configure

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
    Load data from the data path
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


def preprocess_data(train_data:pd.DataFrame,test_data:pd.DataFrame) -> pd.DataFrame:
    """
    preprocess the data 
    
    """
    try:
        logging.info('finding PassengerId column for both train and test file')
        train_data.drop('PassengerId',axis=1,inplace=True)
        test_data.drop('PassengerId',axis=1,inplace=True)
        logging.info('PassengerId column removed')
        logging.info('finding survived column for train file')
        train_data.drop('Survived',axis=1,inplace=True)
        logging.info('Survived column removed')
        return train_data, test_data
    except KeyError:
        logging.error("Column not found")
        raise






def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame, file_path:str) -> None:
    """
    Save data to the file path
    """
    try:
        logging.info('saving train data')
        file_path=os.path.join('data','processed')
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
    train_data, test_data = preprocess_data(train_data, test_data)
    save_data(train_data, test_data, output_path)


if __name__ == "__main__":
    main()