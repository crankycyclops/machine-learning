import collections
import math

class activation:

	# Accepts either a vector (array) or scalar as input and returns the sigmoid.
	def sigmoid(x):

		if type(x) is not str and hasattr(x, "__len__"):

			retVal = []

			for i in range(len(x)):
				retVal.append(1 / (1 + math.exp(-x[i])))

			return retVal

		else:
			return 1 / (1 + math.exp(-x))

	############################################################################

	# Accepts either a vector (array) or scalar as input and returns the
	# hyperbolic tangent.
	def tanh(x):

		if type(x) is not str and hasattr(x, "__len__"):

			retVal = []

			for i in range(len(x)):
				retVal.append(math.tanh(x[i]))

			return retVal

		else:
			print(x)
			return math.tanh(x)

	############################################################################

	# The linear activation function just passes through the original value.
	def linear(x):

		return x

