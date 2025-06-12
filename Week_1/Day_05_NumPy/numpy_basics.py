import numpy as np

# 1. Creating arrays from lists
list_a = [1,2,3,4]
arr_a = np.array(list_a)
print("List:", list_a)
print("Array from list:", arr_a) # Q. What's difference from torch.Tensor and np.array?

# 2. Using NumPy helper functions
zeros = np.zeros((2,3))
print("2*3 zeros array:\n", zeros)

ones = np.ones((3,2))
print("3*2 ones array:\n", ones) # Q. Are there only zeros, ones? Other numbers?

arange = np.arange(0,10,2)
print("Range array (0 to 10, step 2):", arange)

# 3. Element-wise arithmetic
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
print("Addition:", arr1 + arr2)
print("Subtraction:", arr2 - arr1)
print("Multiplication:", arr1 * arr2)
print("Division:", arr2 / arr1)

# 4. Numpy vs Python lists
list1 = [1,2,3]
list2 = [4,5,6]
print("List concatenation:", list1 + list2)
print("Numpy array addition:", np.array(list1)+ np.array(list2))    # Q. How to "reverse" from array to list?
