import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(csv_url):
    data = pd.read_csv(csv_url)
    return data


def print_data(data):
    print('***** Actual data frame *****\n')
    print(data)
    print('\n\n***** First examples *****\n')
    print(data.head())
    print('\n\n***** Data summary *****\n')
    print(data.describe())


def process_data_frame(data):
    data_to_return = map_values(data)
    data_to_return = drop_columns(data_to_return)
    return data_to_return


def map_values(data):
    data['Sex'] = data['Sex'].map({'female' : 1, 'male' : 2})
    return data


def drop_columns(data):
    columns_to_drop = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']
    return data.drop(columns=columns_to_drop)
    

csv_url = 'https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv'
data = process_data_frame(read_data(csv_url))
print_data(data)
