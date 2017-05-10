from numpy import exp, array, random, dot

class NeuralNetwork:
    def __init__(self):
        random.seed(1)


        self.synaptic_weights_l1 = 2 * random.random((3, 10)) - 1
        self.synaptic_weights_l2 = 2 * random.random((10, 10)) - 1
        self.synaptic_weights_l3 = 2 * random.random((10, 1)) - 1


    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, traning_inputs, traning_outputs, traning_interations):
        for iteration in range(traning_interations):

            l1 = self.__sigmoid(dot(traning_inputs, self.synaptic_weights_l1))
            l2 = self.__sigmoid(dot(l1, self.synaptic_weights_l2))
            l3 = self.__sigmoid(dot(l2, self.synaptic_weights_l3))

            #  error
            l3_err = traning_outputs - l3

            l2_err = dot(self.synaptic_weights_l3, l3_err.T) * (self.__sigmoid_derivative(l2).T)
            l1_err = dot(self.synaptic_weights_l2, l2_err) * (self.__sigmoid_derivative(l1).T)

            adjustment3 = dot(l2.T, l3_err)
            adjustment2 = dot(l1.T, l2_err.T)
            adjustment1 = dot(traning_inputs.T, l1_err.T)

            self.synaptic_weights_l1 += adjustment1
            self.synaptic_weights_l2 += adjustment2
            self.synaptic_weights_l3 += adjustment3



    def predict(self, inputs):
        l1 = self.__sigmoid(dot(inputs, self.synaptic_weights_l1))
        l2 = self.__sigmoid(dot(l1, self.synaptic_weights_l2))
        l3 = self.__sigmoid(dot(l2, self.synaptic_weights_l3))

        return l3


# init variables
nr_of_iterations = 10000
traning_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])  # np array
traning_outputs = array([[0, 1, 1, 0]]).T

neural_network = NeuralNetwork()

print("Random stating synaptic weights:")
print(neural_network.synaptic_weights_l1, neural_network.synaptic_weights_l2, neural_network.synaptic_weights_l3)

neural_network.train(traning_inputs, traning_outputs, nr_of_iterations)

print("New synaptic weights after traning")
print(neural_network.synaptic_weights_l1, neural_network.synaptic_weights_l2, neural_network.synaptic_weights_l3)




print("Test new combination: [1,0,0]")
print(neural_network.predict(array([1, 0, 0])))
