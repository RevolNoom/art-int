import numpy
import math
import copy
import random


#    !!!!! A L E R T !!!!!
"""
    imgloader to be updated
    a = np.array([[0],[1],[2]])
    b = np.array([[3],[4],[5],[6]])

    dot(a, b.transpose()) =
    [[0, 0, 0, 0],
     [3, 4, 5, 6],
     [6, 8, 10,12]]

     This property will serve us well in calculating weight changes (i.e. nabla_w)
     That's why, image & label format will see slight but notorious change
     (Discovered through http://neuralnetworksanddeeplearning.com)
"""


"""
    HELPER FUNCTIONS
"""

def quadratic(final_layer_activation, desired_output):
    """ quadratic_cost_function(final_layer_activation, desired_output):
        Takes two numpy arrays argument of same size
        Return the sum of all element-wise (f_l_a - d_o)^2"""
    y = final_layer_activation - desired_output
    return numpy.dot(y, y)

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
    sm = sigmoid(neuron_input)
    return sm * (1 - sm) 
       


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
 
        # !!!!! A L E R T !!!!!
        
        # The existing form of _biases, _weights, _activation, _input have been changed
        # They now wrap each element in a list, 
        # if i.e. self._oldmatrix[x] = i then self._newmatrix[x] = [i]

        self._shape = network_shape
        # _biases[L][n] returns the bias value of neuron 'n' in layer 'L' (n and L are 0-based)
        # randn(shape): Create an array with specified shape, filled with random numbers from Gaussian distribution 
        self._biases = [numpy.random.randn(layer) for layer in self._shape]

        # _weights[L][n2][n1] returns the weight between neuron n2 (in layer L+1) and n1 (in layer L) 
        # No, [n2][n1] wasn't a typo
        self._weights= [numpy.random.randn(self._shape[layer+1], self._shape[layer])
                                                            for layer in range(len(self._shape)-1)]

        
        # The neurons input & activation from the last image feeding
        # Used for learning and accuracy-testing processes
        # None is activated at the beginning
        self._activation= [numpy.zeros([layer, 1]) for layer in self._shape]
        #print("Init activation: \n" + str( self._activation))
        self._input     = copy.deepcopy(self._activation)

        # All the 'd' prefixes are for "derived"
        self._cost   = cost_function[0]
        self._dcost  = cost_function[1]

        self._sigmoid   = sigmoid_function[0]
        self._dsigmoid  = sigmoid_function[1]


    def feed_forward(self, image):
        """
        """
        # Everyone should always be cautious with their data
        if (self._shape[0] != len(image)):
            print("Image has size {0}, different from size of input layer: {1}".format(self._shape[0], len(image)))
            return None
        
        # Feed the first layer with the image
        self._activation[0] = numpy.array(image) / 255

        print("Shape of each activation layer: " + str([act.shape for act in self._activation]))

        ## A L E R T ##

        """Sigmoid input layer 0 has incorrect shape
        """
        # Calculate activations, layer by layer
        for layer in range(len(self._shape)-1):
            #print("Activation Layer {0} shape:\n{1}".format(layer, str(self._activation[layer].shape))) #str(self._activation[layer])))
            print("Weight shape, Activation shape, dot shape:\n {0}, {1}, {2}\nInput shape:\n{3}".format(self._weights[layer].shape, self._activation[layer].shape, numpy.dot(self._weights[layer], self._activation[layer]).shape, self._input[layer].shape)) #str(self._activation[layer])))
            self._input[layer+1]        =   numpy.dot(self._weights[layer], self._activation[layer]) + self._biases[layer+1]
            print("Sigmoid to layer {0}:\nShape: {1}\n{2}".format(layer+1, self._sigmoid(self._input[layer+1]).shape, self._sigmoid(self._input[layer+1])))
            self._activation[layer+1]   =   self._sigmoid(self._input[layer+1])


    def back_propagate(self, data, learning_rate):

        # These matrices are the total changes we need to apply to our current weights and biases
        nabla_b = [numpy.zeros([i]) for i in self._shape]
        nabla_w = [numpy.zeros(L.shape) for L in self._weight]

        # This matrix holds the changes learned in one image
        # temp_nabla_w can be calculated from delta, so no need a whole variable for it
        delta = copy.deepcopy(nabla_b)

        # Start learning
        for d in data:
            self.feed_forward(d[0])
            
            # Calculate the intermediate value dC/dz of the last layer: 
            # (quick reminder: dC/dB = dC/dz. So only dC/dB is done)
            # (See http://neuralnetworksanddeeplearning.com/chap2.html for more details)
            # dC/dB (= dC/dz) = C'(a).sigmoid'(z)
            delta[-1] = self._dcost(self._activation[-1], d[1]) * self._dsigmoid(self._input[-1])

            # Backpropagate to all other layers
            for Layer in range(len(self._shape)-2, 0, -1):
                # Second backpropagation equation:
                # S_L = (w(L+1)T . S_L+1) o sigmoid(L+1)'(z)
                # (reminder, again: dC/dB = S_L)
                # numpy.multiply() or @ does element-wise (i.e. Hadamard). numpy.dot() does dot product.
                delta[Layer] = numpy.dot(self._weights[Layer].transpose(), delta[Layer+1]) * self._dsigmoid(self._input[Layer])

                # Add the result to our nabla
                nabla_b[Layer] += delta[Layer]

                # Fourth equation:
                # dC/dWjk = a(L-1)k * S(L)j
                #print("activation[{0}].transpose:\n{1}".format(Layer-1, self._activation[Layer-1].transpose()))
                nabla_w[Layer] += numpy.dot(delta[Layer], self._activation[Layer-1].transpose())

        # Now, from the nablas, we adjust the network
        for i in range(len(nabla_w)):
            self._weights[i] -= (learning_rate) * nabla_w[i]

        for i in range(len(self._shape)):
            self._biases[i] -= (learning_rate) * nabla_b[i]



    def learn(self, data, learning_rate, batch_size=None, progress_report=False):
        """learn(self, data, learning_rate, batch_size)
            
            data: an array of tuples in the form of (image, label),
                    where 'image' is an array of floats with the same size as layer 0 of the network
                            'label' is an array of floats with the same size as Last layer of the network
                    I suggest using "imgloader" module

            learning_rate: A positive float number. You decide! I'd suggest 1.0 for the first run

            batch_size: If supplied, divide the data into batches of this size (with at most 1 batch in range (batch_size, 2*batch_size))
                        else, it'll just shove all the data into one learning

            progress_report: Report % progress after every finished batch"""

        if batch_size is not None:
            # Using stochastic gradient descent method
            if (batch_size < 1):
                print("You're kidding, right? How am I supposed to divide into batches of size " + str(batch_size) + "?")
                return None

            print("Start learning")
            random.shuffle(data)

            # I come from C++, so please pardon my wariness with python memory allocation
            for i in range(len(data) // batch_size - 1):
                print("Before training:")
                self.test_against(data)
                self.learn(data[i*batch_size : (i+1)*batch_size], learning_rate)
                if progress_report:
                    print("Progress: {0}%".format(i*batch_size/len(data)*100)) 
                    print("Cost function of the last image: {0}".format(self._cost(self._activation[-1], data[(i+1)*batch_size-1][1])))
                print("After training:")
                self.test_against(data)

            print("Whew. Final batch")
            self.learn(data[(len(data) // batch_size -1) * batch_size:], learning_rate)
            print("Learning completed.")

        else:
            self.back_propagate(data, learning_rate)


    def test_against(self, data, deviation_amplitude=0.05, deviation_percentage=5):
        """
            An answer is considered correct if all elements in the answer satisfies:

            result*(100-percentage) - amplitude <= ans <= result*(100+percentage) + amplitude
        """
        correct_answers = 0
        total_tests     = len(data)

        for d in data:
            #ans = self.guess_this_digit(d[0])
            self.feed_forward(d[0])
            ans = self._activation[-1]
            """
            print("Comparing answers:\n{0}".format(d[1]))
            print(["{:4.2f}".format(i) for i in ans])
            ans = [ ans[i] - d[1][i] for i in range(len(ans))]
            if len([i for i in ans if i>0.2]) < 1:
            """
            if len([i for i in range(len(ans)) 
                        if d[1][i][0]*(100-deviation_percentage) - deviation_amplitude <= ans 
                        and ans <= d[1][i][0]*(100+deviation_percentage) + deviation_amplitude]
                        ) == 0:
                ++correct_answers

        print("% accuracy over {0} images: {1}".format(total_tests, correct_answers / total_tests))
