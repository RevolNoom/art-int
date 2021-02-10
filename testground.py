import numpy
import network
import imgloader

n = network.Network([28*28, 30, 10])
data = imgloader.load_data("trainimages", "trainlabels", 10000)

n.learn(data, 0.05, 100, progress_report=True)

test_data = imgloader.load_data("test10Kimages", "test10Klabels", 100)
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
