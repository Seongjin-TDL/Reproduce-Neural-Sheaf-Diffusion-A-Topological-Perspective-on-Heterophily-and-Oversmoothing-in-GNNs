import numpy as np

matrix = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
print("Original matrix:\n", matrix)

# 1. Indexing
print("Element at row 1, col 2:", matrix[1,2])

# 2. Slicing
print("First row:", matrix[0, :])
print("Second column:", matrix[: 1])
print("Top-left 2*2 submatrix:\n", matrix[:2, :2])
print("Down-right 2*2 submatrix:\n", matrix[1:, 1:])

# 3. Reshaping
flat = matrix.reshape(-1)       # I don't fully understand .reshape
print("Flattened:", flat)
reshaped = flat.reshape(9,1)
print("Reshaped to 9*1:", reshaped)

# 4. Transposing
transposed = matrix.T
print("Transposed matrix:\n", transposed)