import csv
import sys
import show
import argparse
import numpy as np
import matplotlib.pyplot as plt

def dataset() :
	try:
		liste = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
	except:
		print('Warning: Failed to load file!\nMake sure data.csv exist.')
		sys.exit(-1)
	m = len(liste)
	x = np.array([1] * m, float)
	nx = np.array([1] * m, float)
	Y = np.array([1] * m, float)
	Theta = np.random.randn(2, 1)
	for i in range(len(liste)) :
		if (i < m):
			x[i] = liste[i][0]
			nx[i] = liste[i][0]
			Y[i] = liste[i][1]
	x = x.reshape(x.shape[0], 1)
	nx = nx.reshape(nx.shape[0], 1)
	Y = Y.reshape(Y.shape[0], 1)
	xmin = np.min(x)
	xmax = np.max(x)
	for i in range(len(nx)):
		nx[i] = (nx[i] - xmin) / (xmax - xmin)
	normX = np.hstack((nx, np.ones(nx.shape)))
	return x, normX, Y, Theta

def model(X, Theta) :
	F = X.dot(Theta)
	return F

def cost_function(m, X, Y, Theta) :
	scal = (1 / 2 * m) * np.sum((model(X, Theta) - Y)**2)
	return scal

def gradient_descent(X, Y, Theta, learning_rate, n_iterations) :
	m = len(Y)
	cost_history = np.array([0] * n_iterations, float)
	for i in range(0, n_iterations) :
		Theta = Theta - learning_rate * (1 / m * X.T.dot(model(X, Theta) - Y))
		cost_history[i] = cost_function(m, X, Y, Theta)
	return Theta, cost_history

def ft_linear_regression() :
	learning_rate = 0.07
	n_iterations = 1000
	x, normX, Y, Theta = dataset()
	final_Theta, cost_history = gradient_descent(normX, Y, Theta, learning_rate, n_iterations)

	prediction = model(normX, final_Theta)
	return x, Y, prediction, cost_history, final_Theta, n_iterations