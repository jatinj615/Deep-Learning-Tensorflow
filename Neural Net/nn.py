from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3,1)) - 1
    
    # activation function
    def __sigmoid(self, x):
        return 1/(1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, num_training_iteration):
        for iteration in range(num_training_iteration):
            output = self.predict(training_set_inputs)

            #calculate the error
            error = training_set_outputs - output

            # multiply the error by the input ad again by the gradient of the sigmoid curve 
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            #adjust the weights
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == '__main__':
    
    # initialise a single neuron neural network
    neural_net = NeuralNetwork()

    print('Random starting synaptic weights:')
    print(neural_net.synaptic_weights)

    # Training Set
    training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    # train neural network suing training set
    neural_net.train(training_set_inputs, training_set_outputs, 10000)

    print('New synaptic weights after training: ')
    print(neural_net.synaptic_weights)

    # test neural network
    print('Predicting [1,0,0] -> ')
    print(neural_net.predict(array([1,0,0])))