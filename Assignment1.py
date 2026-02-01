#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 1
n = float(input("Enter a number : "))
print(n)


# In[4]:


# 2
while True:
    try:
        n = float(input("Enter a number : "))
        print("Valid double value")
        break
    except ValueError :
        print("Invalid double value")


# In[5]:


# 3
array = []
n = int(input("Enter number of elements : "))

for i in range(n):
    element = input("enter the element : ")
    array.append(element)
print(array)


# In[18]:


# 4
n = int(input("Enter n : "))

matrix = []
for i in range(n) :
    row = []
    for j in range(n) :
        row.append(float(input("enter element")))
    matrix.append(row)

transpose = []
for i in range(n) :
    row = []
    for j in range(n) :
        row.append(matrix[j][i])
    transpose.append(row)

print ("Transpose Matrix")
for row in transpose :
    print(row)

row_sum = [0]*n
for i in range(n) : 
    for j in range(n) :
        row_sum[i] += matrix[i][j]

column_sum = [0]*n
for i in range(n) : 
    for j in range(n) :
        column_sum[j] += matrix[i][j]

sum_matrix = [row_sum , column_sum]

print("row sums and column sums")
for row in sum_matrix :
    print(row)


# In[19]:


# 5
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-100, 100, 1000)

# ReLU
relu = np.maximum(0, x)

plt.figure()
plt.plot(x, relu)
plt.title("ReLU Activation Function")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.show()

# Sigmoid and Tanh
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

plt.figure()
plt.plot(x, sigmoid, label="Sigmoid")
plt.plot(x, tanh, label="Tanh")
plt.title("Sigmoid and Tanh Activation Functions")
plt.xlabel("x")
plt.ylabel("Value")
plt.legend()
plt.show()


# In[17]:


# 6
import math

n = int(input("Enter value of n: "))

matrix = []
for i in range(n):
    row = []
    for j in range(n):
        row.append(float(input("enter element")))
    matrix.append(row)

diagonal = [matrix[i][i] for i in range(n)]

exp_values = [math.exp(x) for x in diagonal]
sum_exp = sum(exp_values)

softmax_diag = [x / sum_exp for x in exp_values]

for i in range(n):
    matrix[i][i] = softmax_diag[i]

print("Matrix after applying softmax on diagonal elements : ")
for row in matrix:
    print(row)

# 7
max_diagonal = matrix[0][0]

for i in range(len(matrix)): 
    if matrix[i][i] > max_diagonal :
        max_diagonal = matrix[i][i]

print("Largest element in the principal diagonal : " , max_diagonal)

# 8
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(diagonal, softmax_diag, marker='o', linestyle='-', color='b')
plt.xlabel("Diagonal Elements")
plt.ylabel("Softmax of Diagonal Elements")
plt.title("Softmax vs Principal Diagonal Elements")
plt.grid(True)
plt.show()

