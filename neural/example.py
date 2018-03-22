from NeuralNetwork import VNN

# For now, this is a rather nonsensical example that I'm using just to test that
# I can run values through the VNN class and not run into errors.

# Arbitrary array of input values
data = [1, 2, 3]
nn = VNN(3, 3, nHiddenLayers = 4, nHiddenLayerNeurons = 2)

# Let's hope I don't hit any runtime errors!
print(nn.predict(data))

