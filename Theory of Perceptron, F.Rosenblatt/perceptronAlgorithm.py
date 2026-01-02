import numpy as np

#Defining Inputs and Outputs.
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0,0,0,1])

value = np.zeros(2) #Weights, Initially 0 or random
threshold = 0 #Bias = 0
converged = False #We loop until the output converges to input
error = np.zeros(4) #Initialising the summation
Ypredicted = np.zeros(4)
bias = 0

while not converged:
    error_count = 0
    for index, row in enumerate(X):
        linear_sum = (row[0] * value[0]) + (row[1] * value[1]) + bias
        Ypredicted[index] = 1 if linear_sum > 0 else 0
        error[index] = Y[index] - Ypredicted[index] 
        
        if error[index] != 0:
            value = value + (error[index] * row)
            bias = bias + error[index]
            error_count += 1
            continue
    if error_count == 0:
        converged = True    


print("Output: ", Ypredicted)
print("Final weights: ", value)

