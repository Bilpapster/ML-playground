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


def preprocess_data_frame(data):
    data = map_values(data)
    data = drop_columns(data)
    data = data.dropna()
    return data


def map_values(data):
    data['Sex'] = data['Sex'].map({'female' : 1, 'male' : 2})
    return data


def drop_columns(data):
    columns_to_drop = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']
    return data.drop(columns=columns_to_drop)


def print_data_shape(data, label='data'):
    print('Type of ' + label + ': {}'.format(type(data)))
    print('Shape of ' + label + ': {}'.format(data.shape))
    print(label + ': {}'.format(data))
    

csv_url = 'https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv'
data = preprocess_data_frame(read_data(csv_url))
# print_data(data)

# splits the data to independent variables and the independent variable 'Survived'
feature_names = ['Pclass', 'Sex', 'Age', 'Fare']
x = data[feature_names].values
class_names = ['did not survived', 'survived']
y = data['Survived']

print_data_shape(x)
print()
print_data_shape(y, 'target')
