import numpy
import network
import imgloader

n = network.Network([5, 3, 2])
print("weights: ")
print(n._weights)

print("\n\nbiases: ")
print(n._biases)

print("\n\nActivation: ")
print(n._activation)


n.feed_forward([1, 2, 3, 4, 5])

print("\n\nActivation: ")
print(n._activation)


