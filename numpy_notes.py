# Notes from Keith Galli's video on NumPy (https://www.youtube.com/watch?v=GB9ByFAIAH4)

import numpy as np

# Creating multidimensional array
a = np.array([[1,2,3,4,5,6,7], [8,9,10,11,12,13,14]])
print(a)

# Get 1st row; get 3rd column; then get the 2nd, 4th, and 6th elements of the first row
a[0, :]
a[:, 2]
a[0, 1:6:2]
a[0, 1:6]

# Replace 6th element in the 2nd row with 20
a[1, 5] = 20
print(a)

# Initialize matrix of all 0s
x = np.zeros((2,3))
print(x)

# Initialize matrix of all 1s
x = np.ones((2,2), dtype = 'int32')
print(x)

# Initialize matrix of arbitrary number
x = np.full((2,2), 99)
print(x)

np.full(a.shape, 99)
np.full_like(a, 99)

# Random decimal numbers
np.random.rand(1, 7)

# Random integers 0 through 5
a[0,:] = np.random.randint(6, size = (1,7))

# Create identity matrix
np.identity(5)

# Repeat array
arr = np.array([[1,2,3]])
r1 = np.repeat(arr, 3, axis=0)
print(r1)

# Copying an array (recall that python is pass-by-object-reference which in this case is kinda like pass-by-reference)
a = np.array([1,2,3])
b = a.copy()

# arithmetic
a
a + 2
a - 2
a * 2
a / 2

b = b + 1
a + b

np.cos(a)

# linear algebra
a = np.ones((2,3))
b = np.full((3,2), 2)
np.matmul(a, b)

a = np.array([[1,2], [2,1]])
np.linalg.inv(a)

# stats
stats = np.array([[1,2,3], [4,5,6]])
np.min(stats)
np.max(stats, axis = 1) 
np.sum(stats, axis = 0)

# reshaping arrays
before = np.array([[1,2,3,4], [5,6,7,8]])
after = before.reshape(8,1)
after

row1 = np.array([1,2,3,4])
row2 = np.array([5,6,7,8])
np.vstack([row1, row2])
np.hstack([row1, row2])

# reading in data
import os
os.chdir('home/nick/numpy')

filedata = np.genfromtxt('data.txt', delimiter = ',').astype('int32')
filedata

# boolean masking/advanced indexing
filedata[filedata > 5]
np.any(filedata > 5, axis = 1)
filedata[((filedata > 5) & (filedata < 8))]

