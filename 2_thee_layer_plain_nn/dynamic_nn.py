from numpy import exp, array, random, dot


class NeuralNetwork:
    def __init__(self, no_input_notes, no_output_notes, no_hidden_layers, no_notes_in_hidden_layers):
        random.seed(1)  # random will now give the same random numbers each time the program runs

        # the input layer will take the specifyed number of inputs and pass through the specifyed number of neruns in the hidden layers
        # the hidden layer take the take the specifyed number of neruns in the hidden layers and pass through the specifyed number of neruns in the hidden layers
        # the last layer take the take the specifyed number of neruns in the hidden layers and pass through the specifyed number of neruns in the output

        self.input_layer = 2 * random.random((no_input_notes, no_notes_in_hidden_layers)) - 1

        self.hidden_layers = []

        for layer in xrange(no_hidden_layers):
            self.hidden_layers.append(2 * random.random((no_notes_in_hidden_layers, no_notes_in_hidden_layers)) - 1)

        # the last layer are handled the same way as all the hidden layers,
        # that is why i appended it to the array of hidden layers
        self.hidden_layers.append(2 * random.random((no_notes_in_hidden_layers, no_output_notes)) - 1)

    # converts a number so it will end up between 0 and 1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # gradient of the sigmoid curve
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, traning_inputs, traning_outputs, traning_interations):

        for iteration in xrange(traning_interations):

            #put together input layer and all the hidden layers un one array, and predict the values from the inputs and weights
            layers = []
            layers.append(self.__sigmoid(dot(traning_inputs, self.input_layer)))
            for i in xrange(len(self.hidden_layers)):
                layers.append(self.__sigmoid(dot(layers[-1], self.hidden_layers[i])))

            # calculate the errors
            errors = [] # from last layer to first
            errors.append((traning_outputs - layers[-1]).T)
            for i in reversed(xrange(len(self.hidden_layers))):
                errors.append( dot(self.hidden_layers[i], errors[-1]) * self.__sigmoid_derivative(layers[i]).T)
            errors = errors[::-1] # from first to last

            # adjust the weights (backpropagation)
            self.input_layer += dot(traning_inputs.T, errors[0].T)
            for i in xrange(len(self.hidden_layers)):
                self.hidden_layers[i] += dot(layers[i].T, errors[i + 1].T)


    # predict the outcome py passing the inputs through all the layers
    def predict(self, inputs):
        current_layer = self.__sigmoid(dot(inputs, self.input_layer))

        for i in xrange(len(self.hidden_layers)):
            current_layer = self.__sigmoid(dot(current_layer, self.hidden_layers[i]))

        return current_layer


# init variables
nr_of_iterations = 1000000
traning_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])  # np array
traning_outputs = array([[0, 1, 1, 0]]).T

neural_network = NeuralNetwork(
    no_hidden_layers=3,
    no_notes_in_hidden_layers=25,
    no_input_notes=3,
    no_output_notes=1
)

print len(neural_network.input_layer)
for x in neural_network.hidden_layers:
    print len(x[0])

print "Train"

print "Random stating synaptic weights:"
print neural_network.hidden_layers
#
neural_network.train(traning_inputs, traning_outputs, nr_of_iterations)
#
print "New synaptic weights after traning"
print neural_network.hidden_layers




print "Test new combination: [1,0,0]"
print neural_network.predict(array([1, 0, 0])) # [ 0.99999995]