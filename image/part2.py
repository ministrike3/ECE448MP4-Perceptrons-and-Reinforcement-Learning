import os
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

def euclidean_distance(current_input,checking_against):
    value = 0
    for k in range(0,785):
        value += (current_input[k] - checking_against[k]) ** 2
    return sqrt(value)

def k_nearest_neighbors(current_input,training_data,number_of_neighbors=3):
    distances=[0]*5000
    for i in range(0,5000):
        distances[i]=euclidean_distance(current_input, training_data[i])
    best_list=[]
    best_neighbor_pair=[0,0]
    for i in range(0,number_of_neighbors):
        best_neighbor_pair=[distances.index(min(distances)), min(distances)]
        best_list.append(best_neighbor_pair)
        del distances[distances.index(min(distances))]
    return(best_list)

def make_a_prediction(neighbor_value_tuples,training_labels):
    votes=[0]*10
    for i in neighbor_value_tuples:
        digit=training_labels[int(i[0])]
        votes[digit]+=1
    return(votes.index(max(votes)))

def test_knn_accuracy(testing_data,testing_labels,training_data,training_labels,number_of_neighbors=3):
    accuracy=0
    for test_digit_number in range(0,len(testing_data)):
        get_k_neighbors=k_nearest_neighbors(testing_data[test_digit_number],training_data,number_of_neighbors)
        prediction=make_a_prediction(get_k_neighbors,training_labels)
        if prediction==testing_labels[test_digit_number]:
            accuracy+=1
    print(accuracy/len(testing_data))

if __name__ == "__main__":
    train_data, train_labels = get_training_data()
    test_data, test_labels = get_testing_data()
    #print(euclidean_distance(train_data[0],train_data[1]))
    #print(euclidean_distance(train_data[2], train_data[9]))
    #print(euclidean_distance(train_data[3], train_data[6]))
    #print(k_nearest_neighbors(test_data[0], train_data,5))
    test_knn_accuracy(test_data, test_labels,train_data, train_labels,3)
