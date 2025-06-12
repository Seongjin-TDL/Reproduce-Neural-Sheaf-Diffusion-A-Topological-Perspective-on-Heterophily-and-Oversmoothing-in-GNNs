import numpy as np

# Mean Squared Error (MSE)
true = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
pred = np.array([0.8, 1.6, 1.9, 2.7, 3.2])

error = pred - true
squared_error = error ** 2
mse = np.mean(squared_error)    

print("True values:", true)
print("Predictions:", pred)
print("Error:", error)
print("Squared Error:", squared_error)
print("MSE:", mse)

# One-liner version
print("MSE (one line):", np.mean((pred-true) ** 2))
