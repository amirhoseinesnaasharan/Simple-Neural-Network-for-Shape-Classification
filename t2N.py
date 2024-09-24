import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
cross_samples = np.random.rand(20, 2) - 0.5
circle_samples_outside = np.random.rand(10, 2) + 0.5
circle_samples_inside = np.random.rand(10, 2) - 0.5 - 1.0

circle_samples = np.concatenate(
    [circle_samples_outside, circle_samples_inside])
labels = np.concatenate([np.ones(20), np.zeros(20)])
samples = np.concatenate([cross_samples, circle_samples])

# Initialize neural network parameters
input_size = 2  # Number of input features to the network
# This parameter shows the number of neurons in the hidden layer of the network.
hidden_size = 5
output_size = 1
learning_rate = 0.01
epochs = 10000

# Initialize weights and biases
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Sigmoid activation function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid_derivative


def sigmoid_derivative(x):
    return x * (1 - x)


# Training the neural network
errors = []

for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(samples, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(
        hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Compute error
    error = labels.reshape(-1, 1) - predicted_output
    errors.append(np.mean(np.abs(error)))

    # Backpropagation
    output_error = error * sigmoid_derivative(predicted_output)
    hidden_layer_error = output_error.dot(
        weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(
        output_error) * learning_rate
    bias_output += np.sum(output_error, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += samples.T.dot(hidden_layer_error) * learning_rate
    bias_hidden += np.sum(hidden_layer_error, axis=0,
                          keepdims=True) * learning_rate

# error curve
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Training Error Curve')
plt.show()

# Show the samples and the predicted shapes
plt.scatter(cross_samples[:, 0], cross_samples[:, 1],
            marker='x', label='Cross')
plt.scatter(circle_samples[:, 0],
            circle_samples[:, 1], marker='o', label='Circle')

# Production of test samples
test_samples = np.random.rand(10, 2) - 0.5
test_samples = np.concatenate([test_samples, np.random.rand(10, 2) + 0.5])

# Forward pass for test samples
test_hidden_layer_input = np.dot(
    test_samples, weights_input_hidden) + bias_hidden
test_hidden_layer_output = sigmoid(test_hidden_layer_input)
test_output_layer_input = np.dot(
    test_hidden_layer_output, weights_hidden_output) + bias_output

# Show the predicted shapes
for i in range(len(test_samples)):
    if test_output_layer_input[i] > 0.5:
        plt.scatter(test_samples[i, 0], test_samples[i,
                    1], marker='x', color='red', s=100)
    else:
        plt.scatter(test_samples[i, 0], test_samples[i,
                    1], marker='o', color='blue', s=100)

plt.legend()
plt.title('Predicted Shapes')
plt.show()
