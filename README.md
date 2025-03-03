# Neural Network Visualization and Backpropagation

## Overview
This script implements a simple neural network with forward and backward propagation using the sigmoid activation function. It visualizes the network architecture and activation function and updates weights based on backpropagation.

## Dependencies
```python
import matplotlib.pyplot as plt
import numpy as np
```

## Sigmoid Activation Function
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
```

## Network Visualization
```python
def plot_network():
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis('off')
    ax1.set_title('Neural Network Architecture', pad=20)
```
This function creates a visual representation of the neural network structure.

## Initialize Network Parameters
```python
w1, w2, w3, w4 = 0.15, 0.2, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60
i1, i2 = .05, .10
```

## Forward Pass
```python
neth1 = w1 * i1 + w2 * i2 + b1
neth2 = w3 * i1 + w4 * i2 + b1
outh1 = sigmoid(neth1)
outh2 = sigmoid(neth2)
neto1 = w5 * outh1 + w6 * outh2 + b2
neto2 = w7 * outh1 + w8 * outh2 + b2
o1 = sigmoid(neto1)
o2 = sigmoid(neto2)
```
This section calculates the outputs of the neural network.

## Print Initial Results
```python
print("Initial Outputs:")
print("Final Output 1: {:.4f}".format(o1))
print("Final Output 2: {:.4f}".format(o2))
```

## Backpropagation Implementation
```python
target_o1 = 0.01
target_o2 = 0.99
learning_rate = 0.5

delta_o1 = (o1 - target_o1) * sigmoid_derivative(o1)
delta_o2 = (o2 - target_o2) * sigmoid_derivative(o2)

delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * sigmoid_derivative(outh1)
delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * sigmoid_derivative(outh2)
```
This section calculates the errors at both the output and hidden layers.

## Update Weights and Biases
```python
w5 -= learning_rate * (delta_o1 * outh1)
w6 -= learning_rate * (delta_o1 * outh2)
w7 -= learning_rate * (delta_o2 * outh1)
w8 -= learning_rate * (delta_o2 * outh2)
b2 -= learning_rate * (delta_o1 + delta_o2)

w1 -= learning_rate * (delta_h1 * i1)
w2 -= learning_rate * (delta_h1 * i2)
w3 -= learning_rate * (delta_h2 * i1)
w4 -= learning_rate * (delta_h2 * i2)
b1 -= learning_rate * (delta_h1 + delta_h2)
```

## Forward Pass with Updated Weights
```python
neth1 = w1 * i1 + w2 * i2 + b1
neth2 = w3 * i1 + w4 * i2 + b1
outh1 = sigmoid(neth1)
outh2 = sigmoid(neth2)
neto1 = w5 * outh1 + w6 * outh2 + b2
neto2 = w7 * outh1 + w8 * outh2 + b2
o1 = sigmoid(neto1)
o2 = sigmoid(neto2)
```

## Print Updated Results
```python
print("\nAfter Backpropagation:")
print("Final Output 1: {:.4f}".format(o1))
print("Final Output 2: {:.4f}".format(o2))
```

## Generate Visualizations
```python
plot_network()
plt.show()
```
This final section updates the visualization with new network values.

