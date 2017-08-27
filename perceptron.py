# Perceptron Algorithm on the Sonar Dataset
from random import seed
import sys
from random import randrange
from csv import reader
from multiprocessing import Pool, Array, Process, Manager
import multiprocessing as mp
import numpy as np
import ctypes
import time



shape = 1
_weights = Array(ctypes.c_double, shape, lock=False)
weights = np.frombuffer(_weights, dtype='double').reshape(shape)


_final_scores = Array(ctypes.c_double, shape, lock=False)
final_scores = np.frombuffer(_final_scores, dtype='double').reshape(shape)

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
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
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	global final_scores
	x = []
	#global scores_all
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)

	final_scores = scores
	print(final_scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
	#return scores

# Make a prediction with weights
def predict(row):
	global weights
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	global weights
	#print(weights)
	weights = [0.0 for i in range(len(train[0]))]

	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
	global weights
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction = predict(row)
		predictions.append(prediction)
	return(predictions)



start_time = time.time()


if __name__=='__main__':
	ctx = mp.get_context('fork')

	# Test the Perceptron algorithm on the sonar dataset
	seed(1)
	# load and prepare data
	filename = 'data.csv'
	dataset = load_csv(filename)
	for i in range(len(dataset[0])-1):
		str_column_to_float(dataset, i)
	# convert string class to integers
	str_column_to_int(dataset, len(dataset[0])-1)
	#split data for parallelism
	data_split = cross_validation_split(dataset, 2)
	# evaluate algorithm
	n_folds = 3
	l_rate = 0.01
	n_epoch = 4

	p1 = ctx.Process(target=evaluate_algorithm, args=(data_split[0], perceptron, n_folds, l_rate, n_epoch))
	p2 = ctx.Process(target=evaluate_algorithm, args=(data_split[1], perceptron, n_folds, l_rate, n_epoch))
	#p3 = ctx.Process(target=evaluate_algorithm, args=(data_split[2], perceptron, n_folds, l_rate, n_epoch))
	#p4 = ctx.Process(target=evaluate_algorithm, args=(data_split[3], perceptron, n_folds, l_rate, n_epoch))
	p2.start()
	p1.start()
	#p3.start()
	#p4.start()
	p1.join()
	p2.join()
	#p3.join()
	#p4.join()
	sys.stdout.flush()

	print("--- %s seconds ---" % (time.time() - start_time))

	print('Scores: %s' % final_scores)
	#print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))