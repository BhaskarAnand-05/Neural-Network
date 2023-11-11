import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initial values
I = np.array([0.50, 0.10])
W1 = np.array([[0.15, 0.25], [0.20, 0.30]])
W2 = np.array([[0.35, 0.45], [0.40, 0.50]])
B1 = np.array([0.35, 0.35])
B2 = np.array([0.60, 0.60])
target_outputs = np.array([0.01, 0.99])

# Forward propagation
h_input = np.dot(I, W1.T) + B1
h_output = sigmoid(h_input)

o_input = np.dot(h_output, W2.T) + B2
network_output = sigmoid(o_input)

# Calculate mean squared error loss
loss = np.mean((target_outputs - network_output) ** 2)

# Backpropagation
# Calculate output layer errors and deltas
output_error = target_outputs - network_output
output_delta = output_error * sigmoid_derivative(network_output)

# Calculate hidden layer errors and deltas
hidden_error = np.dot(output_delta, W2)
hidden_delta = hidden_error * sigmoid_derivative(h_output)

# Update weights and biases
learning_rate = 0.5
W2 += learning_rate * np.outer(output_delta, h_output)
B2 += learning_rate * output_delta

W1 += learning_rate * np.outer(hidden_delta, I)
B1 += learning_rate * hidden_delta

# Print the updated weights and biases
print("Updated weights W1:\n", W1)
print("Updated biases B1:\n", B1)
print("Updated weights W2:\n", W2)
print("Updated biases B2:\n", B2)
