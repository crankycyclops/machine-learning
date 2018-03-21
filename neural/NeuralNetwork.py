from __future__ import print_function
from random import random
import pickle

# Vanilla Neural Network
class VNN:

	# nInputNeurons       = How many neurons we can receive input on
	# nOutputNeurons      = How many neurons of output we produce
	# nHiddenLayers       = How many hidden layers lie between the input and the output
	# nHiddenLayerNeurons = How many neurons exist in each hidden layer
	def __init__(self, nInputNeurons, nOutputNeurons,
	nHiddenLayers = 2, nHiddenLayerNeurons = 3):

		# We'll create an array of neurons for each level, each of which will
		# store a computed or assigned value.
		self.neurons = []

		for i in range(0, nHiddenLayers + 2):

			self.neurons[i] = []

			if 0 == i:
				nNeurons = nInputNeurons
			elif nHiddenLayers + 1 == i:
				nNeurons = nOutputNeurons
			else:
				nNeurons = nHiddenLayerNeurons

			# Each neuron will be given a default value of 0.
			for j in range(0, nNeurons):
				self.neurons[i][j] = 0

		# Initialize the neural network with random weights and biases.
		self.neuralParameters = {}
		self.neuralParameters.weights = []
		self.neuralParameters.biases  = []

		for i in range(0, nHiddenLayers + 1):

			self.neuralParameters.weights[i] = []
			self.neuralParameters.biases[i]  = []

			nCols = self.nHiddenLayerNeurons
			if nHiddenLayers == i:
				nCols = self.nOutputNeurons

			for j in range(0, nCols):

				# Populate biases
				self.neuralParameters.biases[i][j] = random()

				# Populate weights
				self.neuralParameters.weights[i][j] = []
				for k in range(0, len(self.neurons[i])):
					self.neuralParameters.weights[i][j][k] = random()

	############################################################################

	# Saves the state of the neural network so it can be restored in the future.
	# This uses the pickle format, and is therefore not secure to load data from
	# untrusted sources. 
	def save(self, filename):

		try:
			pickle.dump({
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
			self.neurons = network.neurons
			self.neuralParameters = network.neuralParameters

		except Exception as e:
			print("Could not save neural network state.")
			print(e.strerror)

