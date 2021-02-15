import numpy
import network
import imgloader

def dtanh(x):
    t = numpy.tanh(x)
    return numpy.ones(t.shape) - t@t

data = imgloader.load_data("trainimages", "trainlabels", 60000)
test_data = imgloader.load_data("test10Kimages", "test10Klabels", 10000)

n1 = network.Network([28*28, 100, 10])
n1.learn(data, 10, 1000, progress_report=True)
n1.test_against(test_data)
