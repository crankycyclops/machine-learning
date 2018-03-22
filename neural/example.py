from NeuralNetwork import VNN

# For now, this is a rather nonsensical example that I'm using just to test that
# I can run values through the VNN class and not run into errors.

# Arbitrary array of input values and their resultant expected predictions. This
# training data, if I'm successful, should help the neural network figure out
# it needs to halve the input values. I'm curious to see how the trained result
# will view combined vectors, like [1, 1, 1].
data = [
	[[1, 0, 0], [0.5, 0, 0]],
	[[0, 1, 0], [0, 0.5, 0]],
	[[0, 0, 1], [0, 0, 0.5]],
	[[2, 0, 0], [1, 0, 0]],
	[[0, 2, 0], [0, 1, 0]],
	[[0, 0, 2], [0, 0, 1]],
	[[4, 0, 0], [2, 0, 0]],
	[[0, 4, 0], [0, 2, 0]],
	[[0, 0, 4], [0, 0, 2]]
]

nn = VNN(3, 3, nHiddenLayers = 8, nHiddenLayerNeurons = 2)

# Let the neural network do its prediction thang!
predicted = nn.predict(data[0][0])

# Let's hope I don't hit any runtime errors!
print('\nOutput neuron values:')
print(predicted)
print('\nExpected neuron values:')
print(data[0][1])
print('\nCost:')
print(nn.computeCost(data[0][1], predicted))
print('\n')

