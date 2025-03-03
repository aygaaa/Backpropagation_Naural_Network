import matplotlib.pyplot as plt
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

def plot_network():
    fig = plt.figure(figsize=(12, 8))
    
    # Network visualization
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis('off')
    ax1.set_title('Neural Network Architecture', pad=20)
    
    circle_radius = 0.15
    positions = {
        'input1': (0.1, 0.7),
        'input2': (0.1, 0.3),
        'hidden1': (0.4, 0.7),
        'hidden2': (0.4, 0.3),
        'output1': (0.7, 0.6),
        'output2': (0.7, 0.4),
        'b1': (0.4, 0.9),
        'b2': (0.7, 0.9)
    }
    
    connections = [
        ('input1', 'hidden1', w1), ('input2', 'hidden1', w2),
        ('input1', 'hidden2', w3), ('input2', 'hidden2', w4),
        ('hidden1', 'output1', w5), ('hidden2', 'output1', w6),
        ('hidden1', 'output2', w7), ('hidden2', 'output2', w8),
        ('b1', 'hidden1', b1), ('b1', 'hidden2', b1),
        ('b2', 'output1', b2), ('b2', 'output2', b2)
    ]
    
    for connection in connections:
        start = positions[connection[0]]
        end = positions[connection[1]]
        weight = connection[2]
        color = 'green' if weight > 0 else 'red'
        linewidth = abs(weight) * 5
        ax1.annotate("", xy=end, xytext=start,
                     arrowprops=dict(arrowstyle="->", color=color, linewidth=linewidth, alpha=0.7))
    
    for node, pos in positions.items():
        color = 'skyblue' if node.startswith('input') else \
                'lightgreen' if node.startswith('hidden') else \
                'gold' if node.startswith('output') else 'gray'
        ax1.add_patch(plt.Circle(pos, circle_radius, color=color, ec='black', lw=2))
        ax1.text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
    
    # Activation function visualization
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.linspace(-5, 5, 100)
    y = sigmoid(x)
    ax2.plot(x, y, lw=2, label='sigmoid activation')
    ax2.set_title('Sigmoid Activation Function')
    ax2.grid(True)
    
    net_values = [neth1, neth2, neto1, neto2]
    for val in net_values:
        ax2.axvline(val, color='r', linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()

# Initialize network parameters
w1, w2, w3, w4 = 0.15, 0.2, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60
i1, i2 = .05, .10

# Forward pass
neth1 = w1 * i1 + w2 * i2 + b1
neth2 = w3 * i1 + w4 * i2 + b1
outh1 = sigmoid(neth1)
outh2 = sigmoid(neth2)
neto1 = w5 * outh1 + w6 * outh2 + b2
neto2 = w7 * outh1 + w8 * outh2 + b2
o1 = sigmoid(neto1)
o2 = sigmoid(neto2)

# Print initial results
print("Initial Outputs:")
print("┌───────────────────────┬───────────────────────┐")
print("│ Final Output 1        │ {:>20.4f} │".format(o1))
print("│ Final Output 2        │ {:>20.4f} │".format(o2))
print("└───────────────────────┴───────────────────────┘")

# Backpropagation implementation
# Set target values and learning rate
target_o1 = 0.01
target_o2 = 0.99
learning_rate = 0.5

# Calculate output layer errors
delta_o1 = (o1 - target_o1) * sigmoid_derivative(o1)
delta_o2 = (o2 - target_o2) * sigmoid_derivative(o2)

# Calculate hidden layer errors
delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * sigmoid_derivative(outh1)
delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * sigmoid_derivative(outh2)

# Calculate weight gradients
# Output layer gradients
d_w5 = delta_o1 * outh1
d_w6 = delta_o1 * outh2
d_w7 = delta_o2 * outh1
d_w8 = delta_o2 * outh2

# Hidden layer gradients
d_w1 = delta_h1 * i1
d_w2 = delta_h1 * i2
d_w3 = delta_h2 * i1
d_w4 = delta_h2 * i2

# Bias gradients
d_b2 = delta_o1 + delta_o2
d_b1 = delta_h1 + delta_h2

# Update weights and biases
w5 -= learning_rate * d_w5
w6 -= learning_rate * d_w6
w7 -= learning_rate * d_w7
w8 -= learning_rate * d_w8
b2 -= learning_rate * d_b2

w1 -= learning_rate * d_w1
w2 -= learning_rate * d_w2
w3 -= learning_rate * d_w3
w4 -= learning_rate * d_w4
b1 -= learning_rate * d_b1

# Forward pass with updated weights
neth1 = w1 * i1 + w2 * i2 + b1
neth2 = w3 * i1 + w4 * i2 + b1
outh1 = sigmoid(neth1)
outh2 = sigmoid(neth2)
neto1 = w5 * outh1 + w6 * outh2 + b2
neto2 = w7 * outh1 + w8 * outh2 + b2
o1 = sigmoid(neto1)
o2 = sigmoid(neto2)

# Print updated results
print("\nAfter Backpropagation:")
print("┌───────────────────────┬───────────────────────┐")
print("│ Final Output 1        │ {:>20.4f} │".format(o1))
print("│ Final Output 2        │ {:>20.4f} │".format(o2))
print("└───────────────────────┴───────────────────────┘")

# Generate visualizations
plot_network()
plt.show()
