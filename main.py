import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import SVM_Linear as svml
import Logistic_Regression as LR


# GLOBAL FUNCTIONS

def dataset_info_gathering(dataset):
    print(dataset.head())
    print(dataset.shape)
    print(dataset.describe())
    # Checking the number of data in each target group
    print(dataset['target'].value_counts())
    # Checking the mean value based on the target variable for people with heart disease and no heart disease
    print(dataset.groupby('target').mean())


def separate_target_column(dataset):
    return [dataset.drop(columns='target', axis=1), dataset['target']]


def split_data_training_testing(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.0198, random_state=2, shuffle=True)
    x_train = data_standardization(x_train)
    x_test = data_standardization(x_test)
    return x_train, x_test, y_train, y_test


def _accuracy_calculation(x, y):
    return accuracy_score(x, y)


def data_standardization(x):
    scalar = StandardScaler()
    scalar.fit(x)
    return scalar.transform(x)


if __name__ == '__main__':
    # Reading CSV dataset file
    cleveland_dataset = pd.read_csv('cleveland_kaggle.csv')

    # Checking dataset statistics
    dataset_info_gathering(cleveland_dataset)

    # Separating output value column from other data
    X, Y = separate_target_column(cleveland_dataset)

    # Making data standard for better machine learning outcome
    # X = data_standardization(X)
    # print("Standard data")
    # print(X)

    # Selecting train and test data from the overall data
    X_train, X_test, Y_train, Y_test = split_data_training_testing(X, Y)

    print("Standard data")
    print("X_train\n", X_train)
    print("X_test\n", X_test)
    print("\n" * 2, "\t" * 10, ">>No feature selection method implemented<<")
    print("\n>> SVM (LINEAR)")
    # SVM (LINEAR)MACHINE LEARNING MODEL IMPLEMENTATION
    # load the SVM classifier and training model accordingly

    svm_classifier = svml.SVM_Linear(X_train, Y_train)

    # Model has been trained
    # Calculating accuracy of the model
    X_train_prediction = svm_classifier.predict_result(X_train)
    X_test_prediction = svm_classifier.predict_result(X_test)

    print('Accuracy of the SVM-Training dataset on the heart disease is: ',
          (_accuracy_calculation(X_train_prediction, Y_train)) * 100, 'percentage')
    print('Accuracy of the SVM-Testing dataset on the heart disease is: ',
          (_accuracy_calculation(X_test_prediction, Y_test)) * 100, 'percentage')

    # print("\nCreated object information")
    # patient_obj = pc.Patient()
    # std_data = scalar.transform(patient_obj.return_as_list())
    # prediction = classifier.predict(std_data)
    #
    # if prediction[0] == 0:
    #     print("\n>>Patient is unlikely to have heart disease.<<\n")
    # else:
    #     print("\n>>Patient is suspected for having HEART DISEASE<<\n")

    # IMPLEMENTATION OF LOGISTIC REGRESSION
    print("\n>> Logistic Regression")
    lr_model = LR.logistic_regression(X_train, Y_train)

    lr_prediction = lr_model.predict_result(X_train)

    lr_train_accuracy = accuracy_score(lr_prediction, Y_train)
    print("Accuracy of Logistic regression for train dataset is: ", lr_train_accuracy * 100, " percent")
    lr_prediction = lr_model.predict_result(X_test)
    lr_train_accuracy = accuracy_score(lr_prediction, Y_test)
    print("Accuracy of Logistic regression for test dataset is: ", lr_train_accuracy * 100, "percent")
