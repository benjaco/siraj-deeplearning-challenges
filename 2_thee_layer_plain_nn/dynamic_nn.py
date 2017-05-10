from numpy import exp, array, random, dot


# benjaco 2017-02-14

class NeuralNetwork:
    def __init__(self, no_input_nodes, no_output_nodes, no_hidden_layers, no_nodes_in_hidden_layers):
        random.seed(1)  # random will now give the same random numbers each time the program runs

        # the input layer will take the specified number of inputs and pass through the specified number of nodes in the hidden layers
        # the hidden layer(s) take the specified number of nodes in the hidden layers and pass through the specified number of nodes in the hidden layers
        # the last layer takes the specified number of nodes in the hidden layers and pass through the specified number of nodes in the output

        self.input_layer = 2 * random.random((no_input_nodes, no_nodes_in_hidden_layers)) - 1

        self.hidden_layers = []

        #   layer=0
        #   while layer <= no_hidden_layers:
        for layer in range(no_hidden_layers):
            self.hidden_layers.append(2 * random.random((no_nodes_in_hidden_layers, no_nodes_in_hidden_layers)) - 1)
            #   layer += 1

        # the last layer are handled the same way as all the hidden layers,
        # that is why i appended it to the array of hidden layers
        self.hidden_layers.append(2 * random.random((no_nodes_in_hidden_layers, no_output_nodes)) - 1)

    # converts a number so it will end up between 0 and 1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # gradient of the sigmoid curve
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, traning_inputs, traning_outputs, traning_interations):
        num_lines = 10
        devider = (traning_interations / num_lines)  # this will gives us 10 lines running through the interations
        #
        #   iteration = 0
        #   while iteration <= traning_interations:
        for iteration in range(traning_interations):
            # {
            # put together input layer and all the hidden layers un one array,
            # and predict the values from the inputs and weights
            layers = []
            layers.append(self.__sigmoid(dot(traning_inputs, self.input_layer)))
            for i in range(len(self.hidden_layers)):
                layers.append(self.__sigmoid(dot(layers[-1], self.hidden_layers[i])))
            #
            # calculate the errors
            errors = []  # from last layer to first
            errors.append((traning_outputs - layers[-1]).T)
            for i in reversed(range(len(self.hidden_layers))):
                errors.append(dot(self.hidden_layers[i], errors[-1]) * self.__sigmoid_derivative(layers[i]).T)
            errors = errors[::-1]  # from first to last
            #
            # adjust the weights (backpropagation)
            self.input_layer += dot(traning_inputs.T, errors[0].T)
            for i in range(len(self.hidden_layers)):
                self.hidden_layers[i] += dot(layers[i].T, errors[i + 1].T)
            #
            if (iteration % devider) == 0:  # the remainder of the devision
                print('traning:', traning_interations, 'iteration:', iteration)
                #   iteration += 1
                # }

    # predict the outcome py passing the inputs through all the layers
    def predict(self, inputs):
        current_layer = self.__sigmoid(dot(inputs, self.input_layer))

        for i in range(len(self.hidden_layers)):
            current_layer = self.__sigmoid(dot(current_layer, self.hidden_layers[i]))

        return current_layer


# init variables
no_feat = 3
no_output = 1
no_hidden_layers = 1
no_nodes_in_hidden_layers = 1
#
no_of_iterations = 1000
traning_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])  # np array
traning_outputs = array([[0, 1, 1, 0]]).T

n_net = NeuralNetwork(no_feat, no_output, no_hidden_layers, no_nodes_in_hidden_layers)

# header
print ("num features:", len(n_net.input_layer))
x = 0
print ('x:', x)

for x in n_net.hidden_layers:
    print ('x[0]:', x[0])
    # print ('len(x[0]:',len(x[0])
    # print ('Layer width: ',str(len(x))])

# body
print ("Train")
# print ("Random stating synaptic weights:")
# print ('NN.hid:',n_net.hidden_layers)
#
n_net.train(traning_inputs, traning_outputs, no_of_iterations)
#
# print ("New synaptic weights after traning")
# print ('n_net.hid:',n_net.hidden_layers)

# footer
print ("Test new combination: [1,0,0]")
print (n_net.predict(array([1, 0, 0])))  # [ 0.99999995]
