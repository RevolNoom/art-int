import numpy
import random

"""
    HELPER FUNCTIONS
"""

def quadratic(final_layer_activation, desired_output):
    """ quadratic_cost_function(final_layer_activation, desired_output):
        Takes two numpy arrays argument of same size
        Return the sum of all element-wise (f_l_a - d_o)^2"""
    y = final_layer_activation - desired_output
    return y @ y

def quadratic_deriv(final_layer_activation, desired_output):
    """ quadratic_deriv(final_layer_activation, desired_output):
        Simply the derivative of quadratic_cost_function."""
    return 2 * (final_layer_activation - desired_output)

def sigmoid(neuron_input):
    """ sigmoid(neuron_input)
        neuron_input is a numpy array, represents the input z of all neurons in a layer
        Return a numpy array of same size, represent the activation of each neuron by the function:
            1.0 / (1.0 + e^(-z))"""
    return 1.0 / (1.0 + numpy.exp(-neuron_input))

def sigmoid_deriv(neuron_input):
    """sigmoid_deriv(neuron_input)
        The derivative of default sigmoid function"""
    ez = numpy.exp(-neuron_input)
    return ez / (1.0 + ez)**2
       


"""
    NETWORK IMPLEMENTATION
"""

class Network:
    """ Future update(?): Option to choose another algorithm other than backpropagation
    """
    def __init__(self, network_shape, 
                #acceptance_threshold = 0.9,
                cost_function = (quadratic, quadratic_deriv), 
                sigmoid_function = (sigmoid, sigmoid_deriv)):
        """  __init__(self, network_shape, 
                        cost_function = (quadratic, quadratic_deriv), 
                        sigmoid_function = (sigmoid, sigmoid_deriv)):
        network_shape: 
            +) An array of positive integers 
            +) network_shape[x] indicates the number of neurons layer x has (layer numbering starts from 0)

       cost_function, sigmoid_function:
            +) Tuples of two functions objects:
                [0] is the function
                [1] is [0]'s derivative

            +) Cost function [0] and [1]: 
                *) Are function objects in the form: c_f(actv, dsro)
                *) Each argument is a numpy array.
                *) 'actv' is the activation of the last layer
                *) 'dsro' is the desired activation of the last layer
                *) Default to quadratic C = (y - a)^2 and its derivative C' = 2(a - y)

            +) Sigmoid function [0] and [1]:
                *) Are function objects in the form: sm(z)
                *) z is a numpy array - the input of each neuron in a layer
                *) Default to 1/(1+e^(-z)) and its derivative e^(-z)/(1+e^(-z))^2"""

        self._layers = len(network_shape)
        self._neurons_of_layer = network_shape
        # _biases[L][i] returns the bias value of neuron 'n' in layer 'L' (n and L are 0-based)
        # Randomize the initial value for fun
        #self._biases = numpy.array([[random.random() for i in range(network_shape[L])]
        #                                                for L in range(len(network_shape))])
        self._biases = [numpy.array([random.random() for i in range(network_shape[layer])]) for layer in range(len(network_shape))]

        # _weights[L][n2][n1] returns the weight between neuron n2 (in layer L+1) and n1 (in layer L) 
        # No, [n2][n1] wasn't a typo
        #self._weights= numpy.array([[[random.random() for n1 in range(network_shape[L])] 
        #                                               for n2 in range(network_shape[L+1])] 
        #                                                    for L in range(len(network_shape)-1)])
        self._weights= [numpy.array([[random.random() for n2 in range(network_shape[L+1])] 
                                                       for n1 in range(network_shape[L])])
                                                            for L in range(len(network_shape)-1)]


        
        # The neurons input & activation from the last image feeding
        # Used for learning and accuracy-testing processes
        # None is activated at the beginning
        self._activation= [numpy.array(numpy.zeros(network_shape[layer])) for layer in range(len(network_shape))]
        self._input     = [numpy.array(numpy.zeros(network_shape[layer])) for layer in range(len(network_shape))]

        # All the 'd' prefixes are for "derived"
        self._cost   = cost_function[0]
        self._dcost  = cost_function[1]

        self._sigmoid   = sigmoid_function[0]
        self._dsigmoid  = sigmoid_function[1]

    def feed_forward(self, image):
        """
        """
        # Everyone should always be cautious with their data
        if (len(self._activation[0]) != len(image)):
            print("Image has size {0}, different from size of input layer: {1}".format(len(self._activation[0]), len(image)))
            return None

        # mat() is deprecating soon
        # Need to tranform this to ndarray
        self._input[0]      = numpy.mat(image)
        self._activation[0] = self._sigmoid(self._input[0])

        for layer in range(len(self._activation)-1):
            self._input[layer+1]        =   numpy.dot(self._activation[layer],  self._weights[layer]) + self._biases[layer+1]
            self._activation[layer+1]   =   self._sigmoid(self._input[layer+1])


    def back_propagate(data, learning_rate):
            # These matrices are the total changes we need to apply to our current weights and biases
            nabla_b = [numpy.array([0 for i in range(network_shape[layer])]) for layer in range(len(network_shape))]
            nabla_w = [numpy.array([[0 for n2 in range(network_shape[L+1])] 
                                                       for n1 in range(network_shape[L])])
                                                            for L in range(len(network_shape)-1)]

            temp_nabla_b = nabla_b
            # We'll calculate temp_nabla_w on the way, so no need to hold a whole variable for it
            # Start learning
            for i in data:
                self.feed_forward(data[0])
                
                # Calculate the intermediate value dC/dz of the last layer:
                # dC/dz = C'(a).sigmoid'(z)
                # (quick reminder: dC/dB = dC/dz)
                # (See http://neuralnetworksanddeeplearning.com/chap2.html for more details)
                temp_nabla_b[-1] = self._dcost(self._activation[-1], data[1]) * self._dsigmoid(self._input[-1])

                # Backpropagate to all other layers
                for Layer in range(self._layers-1, -1, -1):
                    # Second backpropagation equation:
                    # S_L = (wT . sigmoid'(z)) o S_L+1
                    # (reminder, again: dC/dB = S_L)
                    temp_nabla_b[Layer] = numpy.dot(numpy.transpose(self._weights[Layer]), self._dsigmoid(self._input[Layer])) * temp_nabla_b[Layer+1]

                # Back propagation finishes.
                # Add the result to our nablas
                for Layer in range(self._layers):
                    nabla_b[Layer] += temp_nabla_b[Layer]

                for Layer in range(1, self._layers):
                    nabla_w[Layer] +=  self._activation[Layer-1] * temp_nabla_w[Layer]

            # Now, from the nablas, we adjust the network
            for i in range(self._layers-1):
                self._weights[i] -= learning_rate * nabla_w[i]

            for i in range(self._layers):
                self._biases[i] -= learning_rate * nabla_b[i]



    def learn(self, data, learning_rate, batch_size=None):
        """learn(self, data, learning_rate, batch_size)
            
            data: an array of tuples in the form of (image, label),
                    where 'image' is an array of floats with the same size as layer 0 of the network
                            'label' is an array of floats with the same size as Last layer of the network
                    I suggest using "imgloader" module

            learning_rate: A positive float number. You decide! I'd suggest 1.0 for the first run

            batch_size: If supplied, divide the data into batches of this size (with at most 1 batch in range (batch_size, 2*batch_size))
                        else, it'll just shove all the data into one learning"""

        if batch_size is not None:
            # Using stochastic gradient descent method
            if (batch_size < 1):
                print("You're kidding, right? How am I supposed to divide into batches of size " + str(batch_size) + "?")
            random.shuffle(data)

            # I come from C++, so please pardon my wariness with python memory allocation
            for i in range(len(data) // batch_size - 1):
                self.learn(data[i*batch_size : (i+1)*batch_size], learning_rate)
            self.learn(data[(len(data) // batch_size) * batch_size:], learning_rate)

        # Learning implementation here
        else:
            self.back_propagate(data, learning_rate)




        


# These lines are for debugging purpose only
# print(quadratic_cost_function(numpy.array([1, 2, 3, 4]) , numpy.array([0, 1, 2, 3])))
# print(sigmoid_deriv(numpy.array([1,2,3,4])))
