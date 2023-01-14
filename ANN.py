import numpy as np
import pandas as pd
from random import randrange
from random import random
from math import exp


# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


Breast_Can_data = pd.read_csv("./dataset/breast-cancer-wisconsin.data",
                              names=['ID', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'])

Breast_Can_data['x6'] = Breast_Can_data['x6'].replace('?', 0)
Breast_Can_data
Breast_Can_data['x6'] = Breast_Can_data['x6'].astype(int)
avg = Breast_Can_data['x6'].mean()
Breast_Can_data['x6'] = Breast_Can_data['x6'].replace(0, int(avg))
Breast_Can_data['x6'][23]
Breast_Can_data = Breast_Can_data.replace({'x10': {2: 0, 4: 1}})
Breast_Can_data.iloc[:, 1:11].to_csv('data_proc.csv')
training_data_x = Breast_Can_data.iloc[:, 1:11].values
training_data_x

training_data_x.tofile('data.txt')
training_data_y = Breast_Can_data.iloc[:, -1].values
training_data_x = training_data_x.astype(int)
len(training_data_x[0])
training_data_x[0]


def __init__(input_dim, hidden_neurons, output_dim):  # network Initial
    Network_layers = list()
    print('inputdimis: ', input_dim)
    print('output_dim is', output_dim)
    hidden_layer = [{'weights': [random() for i in range(input_dim + 1)]} for i in
                    range(hidden_neurons)]  # weights +1 for bias
    Network_layers.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(hidden_neurons + 1)]} for i in range(output_dim)]
    Network_layers.append(output_layer)
    print(Network_layers)
    return Network_layers


def cal_activation(weights, inputs):
    z = weights[-1]
    # print('length of weights',len(weights))
    for i in range(len(weights) - 1):
        # print(i)
        z = z + weights[i] * inputs[i]  # bias
    return z


def sigmoid(z):
    sigmoid_ = 1.0 / (1.0 + np.exp(-z))
    return sigmoid_


# Sigmoid derivative
def sigmoid_derivative(output):
    return output * (1.0 - output)


def tanh(z):
    tanh_ = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return tanh_


def tanh_derivative(output):
    tanh_derv = 1 - (output) ** 2
    return tanh_derv


def forward_prop(network, record, activation_func):
    inputs = record
    # print('length of inputs', len(inputs))
    for layer in network:
        updated_inputs = []
        for units in layer:
            z = cal_activation(units['weights'], inputs)
            if activation_func == 'sigmoid':
                activation = sigmoid(z)
                # derivative_activation = sigmoid_derivative(activation)
            elif activation_func == 'tanh':
                activation = tanh(z)
                # derivative_activation = tanh_derivative(activation)
            else:
                print('invalid activation entered')
            units['output'] = activation
            updated_inputs.append(units['output'])
        inputs = updated_inputs
    return inputs


def Delta_z(output, activation_func):
    if activation_func == 'sigmoid':
        activation_func_derv = output * (1 - output)
    elif activation_func == 'tanh':
        activation_func_derv = (1 - (output ** 2))
    else:
        print('Invalid activation called')
    return activation_func_derv


def back_prob_error(network, true_value, activation_func):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
            errors.append(neuron['output'] - true_value[j])
        for j in range(len(layer)):
            neuron = layer[j]
    neuron['delta'] = errors[j] * Delta_z(neuron['output'], activation_func)


def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)): neuron['weights'][j] -= learning_rate * neuron['delta'] * inputs[j]
        neuron['weights'][-1] -= learning_rate * neuron['delta']


def train_network(network, training_data, learning_rate, epochs, neuron_outputs, activation_func):
    for epoch in range(epochs):
        error_sum = 0
        for row in training_data:
            # print('row -1 is ',row[-1])
            outputs = forward_prop(network, row, 'sigmoid')
            expected = [0 for i in range(neuron_outputs)]
            # print(expected)
            expected[row[-1]] = 1
            error_sum += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            back_prob_error(network, expected, activation_func)
            update_weights(network, row, learning_rate)
    print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, error_sum,))


def predict(network, row):
    outputs = forward_prop(network, row, 'sigmoid')
    return outputs.index(max(outputs))


def back_propagation(train, test, l_rate, n_epoch, n_hidden, activation_func='sigmoid'):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = __init__(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs, activation_func)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
        fold.append(dataset_copy.pop(index))
    dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def ANN(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for index,unfold_data in enumerate(folds):
        train_set = np.array(folds)
        train_set = np.delete(train_set,index, axis=0)
        train_set = np.concatenate((train_set))
        test_set = list()
        for row in unfold_data:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in unfold_data]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


if __name__ == "__main__":
    n_folds = 5
    l_rate = 0.3
    n_epoch = 500
    n_hidden = 5
    result_score =[]
    for i in range(10):
        intermediate_score = ANN(training_data_x, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
        for item in intermediate_score:
            result_score.append(item)
    print('Scores: %s' % result_score)
    print('Average Accuracy: %.3f%%' % (sum(result_score)/float(len(result_score))))
    print('Standard_deviation = %.3f' % (np.std(result_score)))