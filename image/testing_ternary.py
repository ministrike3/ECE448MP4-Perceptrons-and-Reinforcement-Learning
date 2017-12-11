import os
import matplotlib.pyplot as plt
import numpy as np
from math import *


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
                if character == '+':
                    to_append.append(0.5)
                if character == '#':
                    to_append.append(1)
                if character == '\n':
                    pass
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
                if character == '+':
                    to_append.append(0.5)
                if character == '#':
                    to_append.append(1)
                if character == '\n':
                    pass
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


def create_weights():
    weights = [0] * 10
    for i in range(0, 10):
        weights[i] = [0] * 785
        weights[i][784] = 1
    return (weights)


def train_perceptron_one_epoch(data, labels, weights, epoch_number, alpha_constant=35):
    alpha = (alpha_constant / (alpha_constant + epoch_number))
    for traindigit in range(0, 5000):
        input = data[traindigit]
        correct_label = labels[traindigit]
        possible_activations = [0] * 10
        for possible_class in range(0, 10):
            possible_activations[possible_class] = sum(i[0] * i[1] for i in zip(input, weights[possible_class]))
        guess = possible_activations.index(max(possible_activations))
        if guess != correct_label:
            for feature in range(0, 784):
                weights[guess][feature] -= input[feature] * alpha
                weights[correct_label][feature] += input[feature] * alpha


def post_epoch_test(data, labels, weights):
    accuracy = 0
    for testdigit in range(0, len(data)):
        input = data[testdigit]
        correct_label = labels[testdigit]
        possible_activations = [0] * 10
        for possible_class in range(0, 10):
            possible_activations[possible_class] = sum(i[0] * i[1] for i in zip(input, weights[possible_class]))
        guess = possible_activations.index(max(possible_activations))
        if guess == correct_label:
            accuracy += 1
    print(accuracy / len(data))


def check_digit(input, weights):
    possible_activations = [0] * 10
    for possible_class in range(0, 10):
        possible_activations[possible_class] = sum(i[0] * i[1] for i in zip(input, weights[possible_class]))
    guess = possible_activations.index(max(possible_activations))
    return (guess)


def overall_accuracy(testingData, testingLabels, weights):
    correct = 0
    confusion_matrix = [0] * 10
    for i in range(0, 10):
        confusion_matrix[i] = [0] * 10

    for i in range(0, len(testingData)):
        actual_value = testingLabels[i]
        generated_value = check_digit(testingData[i], weights)
        confusion_matrix[actual_value][generated_value] += 1
        if generated_value == actual_value:
            correct += 1
    correct /= len(testingLabels)
    return (confusion_matrix, correct)


def plot_weights(input_weights):
    x = [i for i in range(0, len(input_weights))]
    y = [i for i in range(0, len(input_weights[0]))]
    intensity = input_weights
    intensity.reverse()
    x, y = np.meshgrid(x, y)
    intensity = np.array(intensity)
    xt = plt.pcolormesh(x, y, intensity)
    plt.colorbar()
    xt.axes.get_xaxis().set_visible(False)
    xt.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    train_data, train_labels = get_training_data()
    test_data, test_labels = get_testing_data()
    # This Happens in a loop per epoch
    weights = create_weights()
    constant=16
    for epoch in range(0,30):
        train_perceptron_one_epoch(train_data, train_labels, weights, epoch, constant)
        post_epoch_test(train_data, train_labels, weights)
    confusion_matrix, overall_probablility = overall_accuracy(test_data, test_labels, weights)
    print(overall_probablility)

    for x in range(0, len(confusion_matrix)):
        row = confusion_matrix[x]
        number_of_this_digit = sum(row)
        for i in range(0, len(row)):
            row[i] = row[i] / number_of_this_digit
        formatted_row = ['%.2f' % elem for elem in row]
        print(formatted_row)
