from NeuralNetwork import VNN

# For now, this is a rather nonsensical example that I'm using just to test that
# I can run values through the VNN class and not run into errors.

# Arbitrary array of input values
data = [1, 2, 3]
nn = VNN(3, 3, nHiddenLayers = 8, nHiddenLayerNeurons = 2)

# The value we're expecting our neural network to compute. This is totally
# arbitrary.
expected = [0, 0, 0.5]

# Let the neural network do its prediction thang!
predicted = nn.predict(data)

# Let's hope I don't hit any runtime errors!
print('\nOutput neuron values:')
print(predicted)
print('\nExpected neuron values:')
print(expected)
print('\nCost:')
print(nn.computeCost(expected, predicted))
print('\n')

