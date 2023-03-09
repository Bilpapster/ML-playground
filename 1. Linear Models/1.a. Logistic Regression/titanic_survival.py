import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from collections import Counter

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
    

def my_plot(data):
    x_axis = 2
    y_axis = 3

    plt.figure(1, figsize=(8, 6))
    plt.clf()

    # plot the selected features, with the target used to color the points, and the cmap used to define the color map
    plt.scatter(data[:, x_axis], data[:, y_axis], c=y, cmap=plt.cm.Set2, edgecolor='k') 

    # add axis labels
    plt.xlabel(feature_names[x_axis]) # Label the x-axis with the name of the feature used for the x-axis
    plt.ylabel(feature_names[y_axis]) # Label the y-axis with the name of the feature used for the y-axis

    plt.show()


def get_logistic_regression_classifier():
    return LogisticRegression(max_iter=300, solver='lbfgs', multi_class='auto', verbose=0)


def make_predictions(model):
    #                  'Pclass', 'Sex', 'Age',  'Fare'
    x_new = np.array([[2,          2,    25,    77.77]]) # It has to be 2D
    prediction = model.predict(x_new)[0]
    print('Prediction: {}'.format(prediction))
    print('Predicted target name: {}'.format(class_names[prediction]))

    #                  'Pclass', 'Sex', 'Age',  'Fare'
    x_new = np.array([[1,         1,     27,    155.55]]) # It has to be 2D
    prediction = model.predict(x_new)[0]
    print("Prediction: {}".format(prediction))
    print("Predicted target name: {}".format(class_names[prediction]))


def make_manual_splitting(x, y, train_percentage=0.7):
    nof_training_examples = np.int32(np.ceil(len(x) * train_percentage))
    x_train = x[:nof_training_examples]
    x_test = x[nof_training_examples:]
    y_train = y[:nof_training_examples]
    y_test = y[nof_training_examples:]

    print(Counter(y_train))
    print()
    print(Counter(y_test))


def make_sklearn_splitting(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=6)
    print(Counter(y_train))
    print()
    print(Counter(y_test))


def sklearn_train_and_predict(x, y):
    # split train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=6)

    # construct model
    classifier = LogisticRegression(max_iter=300, solver='lbfgs', multi_class='auto', verbose=0)
    model = classifier.fit(x_train, y_train)

    #predict
    y_predicted = model.predict(x_test)

    # evaluate (manually)
    print('Test set score: {:.2f}'.format(np.mean(y_predicted == y_test)))

    # evaluate (with scikit-learn)
    accuracy = accuracy_score(y_test, y_predicted)
    print('Accuracy score: {:.2f}'.format(accuracy))

    # print classification report with scikit-learn
    report = classification_report(y_test, y_predicted)
    print('Classification report (using scikit-learn)\n', report)


csv_url = 'https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv'
data = preprocess_data_frame(read_data(csv_url))
# print_data(data)

# splits the data to independent variables and the independent variable 'Survived'
feature_names = ['Pclass', 'Sex', 'Age', 'Fare']
x = data[feature_names].values
class_names = ['did not survived', 'survived']
y = data['Survived']

print_data(data)
print()
print_data_shape(x)
print()
print_data_shape(y, 'target')
print()
my_plot(x)
classifier = get_logistic_regression_classifier()
model = classifier.fit(x, y)
make_predictions(model)
print()
make_manual_splitting(x, y, 0.7)
make_sklearn_splitting(x, y)
sklearn_train_and_predict(x, y)
