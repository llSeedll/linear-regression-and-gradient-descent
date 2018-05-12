from math import sqrt
from csv import reader
from random import randrange
from random import seed

def load_data(filename):
	data_set = list()
	with open(filename, 'r') as f:
		file_reader = reader(f, delimiter=',')
		for row in file_reader:
			if not row:
				continue
			data_set.append(row)
	#split row value with commas
	#[ col.split(',') for row in data_set for col in row]
	return data_set

def preprocess(data_set, col):
	for row in data_set:
		row[col] = float(row[col].strip())

def split_dataset(data_set, keep_rate=.8):
	test_set = list()
	test_size = (1.0 - keep_rate) * len(data_set)
	train_set = list(data_set)
	while len(test_set) < test_size:
		index = randrange(len(train_set))
		test_set.append(train_set.pop(index))
	return train_set, test_set

def mean(data):
	return sum(data) / float(len(data))

def variance(data, mean):
	return sum([(x - mean)**2 for x in data])

def covariance(X, Y, mean_x, mean_y):
	return sum([(X[i] - mean_x) * (Y[i] - mean_y) for i in range(len(X))])

def mean_squared_error(correct, predicted):
	count = len(correct)
	total_error = sum([(predicted[i] - correct[i])**2 for i in range(count)])
	return sqrt(total_error / float(count))

def coefficients(data_set):
	X = [data[0] for data in data_set]
	Y = [data[1] for data in data_set]
	x_mean, y_mean = mean(X), mean(Y)
	m = covariance(X, Y, x_mean, y_mean) / variance(X, x_mean)
	b = y_mean - (m * x_mean)
	return [m, b]

def simpleLR(train_set, test_set):
	m, b = coefficients(train_set)
	predictions = [predict(x[0], m, b) for x in test_set]
	return predictions, m, b

def predict(x, m, b):
	return (m * x) + b

def evaluate(data_set, algorithm, keep_rate=.8):
	train_set, test_set = split_dataset(data_set, keep_rate)
	test_data = list()
	for row in test_set:
		row_data = list(row)
		row_data[-1] = None
		test_data.append(row_data)
	predicted, m, b = algorithm(train_set, test_data)
	correct = [row[-1] for row in test_set]
	error = mean_squared_error(correct, predicted)
	return error, m, b

if __name__ == '__main__':
	# train_size = 72%
	# test_size  = 100% - 72% = 28%
	keep_rate = .72
	#seed(1)
	filename = 'data.csv'
	data_set = load_data(filename)
	[preprocess(data_set, i) for i in range(len(data_set[0]))]
	error, m, b = evaluate(data_set, simpleLR, keep_rate)
	predicted = predict(30, m, b)
	print('Mean Squared Error: %.5f' % (error))
	print('Prediction: %.5f [ %.5f - %.5f ]' % (predicted, predicted - error, predicted + error))
