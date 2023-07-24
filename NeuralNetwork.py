import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Read the MNIST dataset from a CSV file using pandas.
df = pd.read_csv("train.csv")

# Convert the dataset into a numpy array for easier manipulation.
df = np.array(df)
m, n = df.shape  # m is the number of rows, and n is the number of columns with the label column
np.random.shuffle(df)  # Shuffle the data to avoid any order bias before splitting into dev and training sets

# Split the data into development and training sets
df_dev = df[0:1000].T
Y_dev = df_dev[0]  # Labels for the development set
X_dev = df_dev[1:n]  # Features for the development set
X_dev = X_dev / 255.  # Normalize the feature data by dividing by 255 to scale it between 0 and 1.

df_train = df[1000:m].T
Y_train = df_train[0]  # Labels for the training set
X_train = df_train[1:n]  # Features for the training set
X_train = X_train / 255.  # Normalize the feature data.

_, m_train = X_train.shape  # m_train is the number of training examples.

# He initialization function for initializing the weights.
def He_init(shape):
    return np.random.randn(*shape) * np.sqrt(2.0 / shape[1])

# Initialize the parameters (weights and biases) for each layer of the neural network.
def init_params():
    W1 = He_init((20, 784))  # 20 nodes in the first hidden layer, 784 input features (pixels of 28x28 image)
    b1 = np.zeros((20, 1))   # Bias for the first hidden layer
    W2 = He_init((20, 20))   # 20 nodes in the second hidden layer, connected to the first hidden layer
    b2 = np.zeros((20, 1))   # Bias for the second hidden layer
    W3 = He_init((20, 20))   # Third hidden layer with 20 nodes
    b3 = np.zeros((20, 1))   # Bias for the third hidden layer
    W4 = He_init((20, 20))   # Fourth hidden layer with 20 nodes
    b4 = np.zeros((20, 1))   # Bias for the fourth hidden layer
    W5 = He_init((10, 20))   # Output layer with 10 nodes (representing 10 classes)
    b5 = np.zeros((10, 1))   # Bias for the output layer

    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5

# ReLU activation function.
def ReLU(Z):
    return np.maximum(Z, 0)

# Softmax activation function for the output layer.
def Softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / np.sum(e_Z, axis=0, keepdims=True)
    return A

# Forward propagation through the neural network.
def forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = ReLU(Z3)
    Z4 = W4.dot(A3) + b4
    A4 = ReLU(Z4)
    Z5 = W5.dot(A4) + b5
    A5 = Softmax(Z5)
    return Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5

# Convert labels to one-hot representation for multiclass classification.
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Derivative of ReLU activation function for backpropagation.
def deriv_ReLU(Z):
    return Z > 0

# Backward propagation to compute gradients for each layer's parameters.
def backward_prop(Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, W1, W2, W3, W4, W5, X, Y):
    one_hot_Y = one_hot(Y)
    dZ5 = A5 - one_hot_Y
    dW5 = 1 / m * dZ5.dot(A4.T)
    db5 = 1 / m * np.sum(dZ5, axis=1, keepdims=True)
    dZ4 = W5.T.dot(dZ5) * deriv_ReLU(Z4)
    dW4 = 1 / m * dZ4.dot(A3.T)
    db4 = 1 / m * np.sum(dZ4, axis=1, keepdims=True)
    dZ3 = W4.T.dot(dZ4) * deriv_ReLU(Z3)
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5

# Update parameters using gradient descent.
def update_params(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    W4 -= alpha * dW4
    b4 -= alpha * db4
    W5 -= alpha * dW5
    b5 -= alpha * db5
    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5

# Get predictions by selecting the index of the maximum value from the output (softmax) layer.
def get_predictions(A5):
    return np.argmax(A5, axis=0)

# Calculate the accuracy of the predictions.
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent training process.
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, X)
        dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5 = backward_prop(Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5, W1, W2, W3, W4, W5, X, Y)
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = update_params(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, dW1, db1, dW2, db2, dW3, db3, dW4, db4, dW5, db5, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A5)
            print("Accuracy", get_accuracy(predictions, Y))
    return W1, b1, W2, b2, W3, b3, W4, b4, W5, b5

# Train the model using gradient descent.
W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = gradient_descent(X_train, Y_train, 0.1, 500)

# Function to make predictions using the trained model.
def make_predictions(X, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5):
    _, _, _, _, _, _, _, _, _, A5 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, X)
    predictions = get_predictions(A5)
    return predictions

# Function to test individual predictions and display results.
def test_prediction(index, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Test some individual predictions using the trained model.
test_prediction(0, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
test_prediction(1, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
test_prediction(2, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
test_prediction(3, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
test_prediction(4, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
test_prediction(5, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
test_prediction(6, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
test_prediction(7, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
test_prediction(8, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
test_prediction(9, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
test_prediction(10, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
