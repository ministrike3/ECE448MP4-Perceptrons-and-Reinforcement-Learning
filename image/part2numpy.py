import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

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

# Much of this code is either from
# or inspired by
#https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/

# Our own implementation of k nearest neighbors is provided in the part2.py file, but was much too slow to reasonably
#be able to test parameters (4-5 minutes per run)

# This implementation used numpy to seriously speed things up
def predict(X_train, y_train, x_test, k):
    # create list for distances and targets
    distances = []
    targets = []

    for i in range(len(X_train)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        # add it to list of distances
        distances.append([distance, i])

    # sort the list
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    # return most common target
    return Counter(targets).most_common(1)[0][0]


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

def plot_error(accuracies):
    neighbors=[i for i in range(0,len(accuracies))]
    plt.plot(neighbors, accuracies)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Accuracy')
    plt.show()

def plot_time(time_array):
    neighbors=[i for i in range(0,len(time_array))]
    plt.plot(neighbors, time_array)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('RunTime')
    plt.show()


if __name__ == "__main__":
    train_data, train_labels = get_training_data()
    test_data, test_labels = get_testing_data()
    train_data, train_labels = get_training_data()
    test_data, test_labels = get_testing_data()
    train_data = np.array([np.array(xi) for xi in train_data])
    train_labels = np.array([np.array(xi) for xi in train_labels])
    test_data = np.array([np.array(xi) for xi in test_data])
    test_labels = np.array([np.array(xi) for xi in test_labels])
    ac=[]
    time_array=[]

    #Run these statements once to get the best value of k and the graphs needed
    # for k in range(1,21):
    #     start = time.time()
    #
    #     predictions = []
    #     for i in range(len(test_data)):
    #         predictions.append(predict(train_data, train_labels, test_data[i, :], k))
    #     confusion_matrix, accuracy = accuracy_score(test_labels, predictions)
    #
    #     end = time.time()
    #     print(end - start)
    #     print(accuracy)
    #
    #     ac.append(accuracy)
    #     time_array.append(end - start)
    #
    # plot_error(ac)
    # plot_time(time_array)

    #Run this stuff once we've isolated the best k for more information
    start = time.time()
    predictions = []
    for i in range(len(test_data)):
        predictions.append(predict(train_data, train_labels, test_data[i, :], 3))

    confusion_matrix, accuracy = accuracy_score(test_labels, predictions)

    print(accuracy)
    end = time.time()
    print(end - start)
    for x in range(0, len(confusion_matrix)):
        row = confusion_matrix[x]
        number_of_this_digit = sum(row)
        for i in range(0, len(row)):
            row[i] = row[i] / number_of_this_digit
        formatted_row = ['%.2f' % elem for elem in row]
        print(formatted_row)
