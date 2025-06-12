import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# (XOR)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])


input_size = 2  
hidden_size = 10  
output_size = 1  
learning_rate = 0.5
epochs = 10000

# pesos y sesgos
np.random.seed(42)
W1 = np.random.rand(input_size, hidden_size)
b1 = np.random.rand(hidden_size)
W2 = np.random.rand(hidden_size, output_size)
b2 = np.random.rand(output_size)


errors = []

# Training
for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(X_train, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    output = sigmoid(output_layer_input)
    
    # (MSE)
    error = y_train - output
    mse = np.mean(error**2)
    errors.append(mse)
    
    # Backpropagation
    output_gradient = error * sigmoid_derivative(output)
    hidden_gradient = np.dot(output_gradient, W2.T) * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    W2 += np.dot(hidden_layer_output.T, output_gradient) * learning_rate
    b2 += np.sum(output_gradient, axis=0) * learning_rate
    W1 += np.dot(X_train.T, hidden_gradient) * learning_rate
    b1 += np.sum(hidden_gradient, axis=0) * learning_rate
    
    
    if epoch % 1000 == 0:
        print(f"Epoca {epoch}, Error MSE: {mse}")


plt.plot(errors)
plt.xlabel("Epocas")
plt.ylabel("MSE")
plt.title("Convergencia del Error MSE")
plt.show()


print("\nResultados de prueba después del entrenamiento:")
for i in range(len(X_train)):
    hidden_layer_input = np.dot(X_train[i], W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    prediction = sigmoid(output_layer_input)
    print(f"Entrada: {X_train[i]}, Salida esperada: {y_train[i]}, Predicción: {prediction}")