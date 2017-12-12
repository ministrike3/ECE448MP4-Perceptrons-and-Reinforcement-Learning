import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import *

def get_training_data():
    digit_dir = os.getcwd() + "/trainingData/"
    training_images = digit_dir + "trainingimages"
    new_text = open(training_images, "r")
    train_data = []
    to_append = []
    line_counter = 0
    for line in new_text.readlines():
        if line_counter != 28:
            line_counter += 1
            for character in line:
                if character == ' ':
                    to_append.append(0)
                elif character == '\n':
                    pass
                else:
                    to_append.append(1)
        if line_counter == 28:
            line_counter = 0
            to_append.append(1)
            train_data.append(to_append)
            to_append = []
    training_labels = digit_dir + "traininglabels"
    new_labels = open(training_labels, "r")
    labels = []
    for line in new_labels.readlines():
        labels.append(int(line))
    return train_data, labels


def get_testing_data():
    digit_dir = os.getcwd() + "/testData/"
    testing_images = digit_dir + "testimages"
    new_text = open(testing_images, "r")
    train_data = []
    to_append = []
    line_counter = 0
    for line in new_text.readlines():
        if line_counter != 28:
            line_counter += 1
            for character in line:
                if character == ' ':
                    to_append.append(0)
                elif character == '\n':
                    pass
                else:
                    to_append.append(1)
        if line_counter == 28:
            line_counter = 0
            to_append.append(1)
            train_data.append(to_append)
            to_append = []
    training_labels = digit_dir + "testlabels"
    new_labels = open(training_labels, "r")
    labels = []
    for line in new_labels.readlines():
        labels.append(int(line))
    return train_data, labels


def accuracy_score(test_labels,predictions):
    accuracy=0
    confusion_matrix=[0]*10
    for i in range(len(confusion_matrix)):
        confusion_matrix[i]=[0]*10
    for i in range(0,len(test_labels)):
        confusion_matrix[test_labels[i]][predictions[i]] += 1
        if test_labels[i]==predictions[i]:
            accuracy+=1
    return(confusion_matrix,accuracy/len(test_labels))


if __name__ == "__main__":
    train_data, train_labels = get_training_data()
    test_data, test_labels = get_testing_data()
    # This Happens in a loop per epoch
    clf = RandomForestClassifier()
    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    confusion_matrix, accuracy = accuracy_score(test_labels, predictions)

    print(accuracy)
    for x in range(0, len(confusion_matrix)):
        row = confusion_matrix[x]
        number_of_this_digit = sum(row)
        for i in range(0, len(row)):
            row[i] = row[i] / number_of_this_digit
        formatted_row = ['%.2f' % elem for elem in row]
        print(formatted_row)