import numpy
import network
import imgloader

def dtanh(x):
    t = numpy.tanh(x)
    return numpy.ones(t.shape) - t@t

#n = network.Network([28*28, 100, 50, 10], sigmoid_function=(numpy.tanh, dtanh) )
n = network.Network([28*28, 100, 50, 10])
data = imgloader.load_data("trainimages", "trainlabels", 1000)

n.learn(data, 0.1, 100, progress_report=True)

test_data = imgloader.load_data("test10Kimages", "test10Klabels", 100)
#test_data = numpy.random.choice(data, 100)
n.test_against(test_data)
"""print("weights: ")
print(n._weights)

print("\n\nbiases: ")
print(n._biases)

print("\n\nActivation: ")
print(n._activation)


#n.feed_forward([1, 2, 3, 4, 5])

#print("\n\nActivation: ")
#print(n._activation)"""

#n.learn([[1,2,3,4,5],[0.0, 1.0]], 0.5)
