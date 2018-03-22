from __future__ import print_function

from random import random
import pickle
import numpy as np

from NeuralMath import activation

# Vanilla Neural Network
class VNN:

	# nInputNeurons       = How many neurons we can receive input on
	# nOutputNeurons      = How many neurons of output we produce
	# nHiddenLayers       = How many hidden layers lie between the input and the output
	# nHiddenLayerNeurons = How many neurons exist in each hidden layer
	# activation          = Which activation function to use on each neuron's value
	def __init__(self, nInputNeurons, nOutputNeurons,
	nHiddenLayers = 2, nHiddenLayerNeurons = 3, activationName = 'tanh'):

		if 'tanh' == activationName:
			self.activation = activation.tanh
		elif 'sigmoid' == activationName:
			self.activation = activation.sigmoid
		elif 'linear' == activationName:
			self.activation = activation.linear
		else:
			raise Exception("Supported activation functions include 'tanh', 'sigmoid' and 'linear'.")

		# We'll create an array of neurons for each level, each of which will
		# store a computed or assigned value.
		self.neurons = []

		for i in range(0, nHiddenLayers + 2):

			self.neurons.append([])

			if 0 == i:
				nNeurons = nInputNeurons
			elif nHiddenLayers + 1 == i:
				nNeurons = nOutputNeurons
			else:
				nNeurons = nHiddenLayerNeurons

			# Each neuron will be given a default value of 0.
			for j in range(0, nNeurons):
				self.neurons[i].append(0)

		# Initialize the neural network with random weights and biases.
		self.neuralParameters = {}
		self.neuralParameters['weights'] = []
		self.neuralParameters['biases']  = []

		for i in range(0, nHiddenLayers + 1):

			self.neuralParameters['weights'].append([])
			self.neuralParameters['biases'].append([])

			nCols = nHiddenLayerNeurons
			if nHiddenLayers == i:
				nCols = nOutputNeurons

			for j in range(0, nCols):

				# Populate biases
				self.neuralParameters['biases'][i].append(random())

				# Populate weights
				self.neuralParameters['weights'][i].append([])
				for k in range(0, len(self.neurons[i])):
					self.neuralParameters['weights'][i][j].append(random())

	############################################################################

	# Saves the state of the neural network so it can be restored in the future.
	# This uses the pickle format, and is therefore not secure to load data from
	# untrusted sources. 
	def save(self, filename):

		try:
			pickle.dump({
				'activation': self.activation,
				'neurons': self.neurons,
				'neuralParameters': self.neuralParameters
			}, filename)

		except Exception as e:
			print("Could not save neural network state.")
			print(e.strerror)

	############################################################################

	# Restores the state of the neural network from a previously saved instance.
	# This uses the pickle format, and is therefore not secure to load data from
	# untrusted sources. 
	def restore(self, filename):

		try:
			network = pickle.load(filename)
			self.activation = network.activation
			self.neurons = network.neurons
			self.neuralParameters = network.neuralParameters

		except Exception as e:
			print("Could not save neural network state.")
			print(e.strerror)

	############################################################################

	# Takes as input an array of values (should match the number of input
	# neurons) and outputs a probability distribution based on the final
	# output neuron values.
	def predict(self, data):

		if len(data) != len(self.neurons[0]):
			raise Exception('Number of data elements must match the number of input neurons!')

		# Assign data to the input layer neurons.
		for i in range(0, len(self.neurons[0])):
			self.neurons[0][i] = data[i]

		# Next, feed forward until we reach the output neurons.
		for i in range(1, len(self.neurons)):
			self.neurons[i] = np.dot(self.neuralParameters['weights'][i - 1], self.neurons[i - 1])
			self.neurons[i] = np.add(self.neuralParameters['biases'][i - 1], self.neurons[i])
			self.neurons[i] = self.activation(self.neurons[i])

		# Finally, return the values of the output neurons.
		return self.neurons[len(self.neurons) - 1]

	############################################################################

	# Takes as input an ideal set of output neuron values (goodData) and an
	# actual set of computed neuron values (badData) and calculates the cost,
	# which is a mathematical way of expressing how much the actual values
	# deviate from the expected values.
	#
	# When running a lot of training examples through the network as part of a
	# single test set, the average cost should be computed, which is the average
	# of all computeCost() results.
	def computeCost(self, goodData, badData):

		if len(goodData) != len(badData):
			raise Exception('goodData and badData must have the same number of elements.')

		cost = 0

		for i in range(0, len(goodData)):
			cost += (badData[i] - goodData[i]) ** 2

		return cost

