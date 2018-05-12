import numpy as np
import pandas
import math

def evaluate(x, m, b):
	'''
	y = m*x + b
	m = slope
	b = y-intercept
	'''
	return (m*x) + b

def d_m(point, m, b, N):
	'''
	Derivative of mx + b with respect to m
	'''
	x = point[0]
	y = point[1]
	return -(2/N) * x * (y - ((m * x) + b))

def d_b(point, m, b, N):
	'''
	Derivative of mx + b with respect to m
	'''
	N = float(N)
	x = point[0]
	y = point[1]
	return -(2/N) * (y - ((m * x) + b))

def computeError(m, b, points, N):
	'''
	
	'''
	totalError = 0
	for _, point in points.iterrows():
		x = point[0]
		y = point[1]
		y_hat = evaluate(x, m, b)
		#if not math.isnan(y_hat):
		#	print(y_hat, m, x, b)
		totalError += (y - y_hat)**2
	return totalError / float(N)

def step(m, b, points, learning_rate, N):
	m_gradient = 0
	b_gradient = 0
	#loss = computeError(m, b, points, N)
	for _, point in points.iterrows():
		m_gradient += d_m(point, m, b, N)
		b_gradient += d_b(point, m, b, N)
	m -= m_gradient * learning_rate
	b -= b_gradient * learning_rate
	return [m, b]

def gradientDescent(points, initial_m, initial_b, learning_rate, iterations, N):
	m = initial_m
	b = initial_b
	for i in range(iterations):
		m, b = step(m, b, points, learning_rate, N)
		if (i%100 == 0):
			loss = computeError(m, b, points, N)
			print("{} iterations: m={}, b={}, error={}".format(i, m, b, loss))
	return [m, b]

def run(data, learning_rate, iterations, N):
	m = 0
	b = 0
	print("Started at m={}, b={}, error={}".format(m, b, computeError(m, b, data, N)))
	[m, b] = gradientDescent(data, m, b, learning_rate, iterations, N)
	print("Ended after {} iterations with m={}, b={}, error={}".format(iterations, m, b, computeError(m, b, data, N)))

if __name__ == '__main__':
	file = 'data.csv'
	learning_rate = 1e-4
	iterations = 1000
	data = np.genfromtxt(file, delimiter=',')
	N = len(data)
	data = pandas.DataFrame(data)
	run(data, learning_rate, iterations, N)
