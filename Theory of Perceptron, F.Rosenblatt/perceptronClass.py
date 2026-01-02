import numpy as np

class ModernPerceptron: #Uses modern terms.
    def __init__(self, input_dim=2, learning_rate=1.0):
        self.weights = np.zeros(input_dim)  # Weights, initially zeros
        self.bias = 0.0  # Bias, initially zero
        self.learning_rate = learning_rate

    def fit(self, X, y):
        converged = False
        n_samples = X.shape[0]
        errors = np.zeros(n_samples)
        y_pred = np.zeros(n_samples)
        while not converged:
            error_count = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred[idx] = 1 if linear_output > 0 else 0
                errors[idx] = y[idx] - y_pred[idx]
                if errors[idx] != 0:
                    self.weights += self.learning_rate * errors[idx] * x_i
                    self.bias += self.learning_rate * errors[idx]
                    error_count += 1
            if error_count == 0:
                converged = True
        return self.weights, y_pred

class PerceptronLegacyTerms:        #Uses terms used by rosenblatt on his paper.
    def __init__(self):
        self.value = np.zeros(2) #Weights, Initially 0 or random
        self.threshold = 0 #Bias = 0


    def converge(self,X, Y):
        converged = False #We loop until the output converges to input
        error = np.zeros(4) #Initialising the summation
        Ypredicted = np.zeros(4)
        while not converged:
            error_count = 0
            for index, row in enumerate(X):
                linear_sum = (row[0] * self.value[0]) + (row[1] * self.value[1]) + self.threshold
                Ypredicted[index] = 1 if linear_sum > 0 else 0
                error[index] = Y[index] - Ypredicted[index] 
                
                if error[index] != 0:
                    self.value = self.value + (error[index] * row)
                    self.threshold = self.threshold + error[index]
                    error_count += 1
                    continue
            if error_count == 0:
                converged = True    
        return self.value, Ypredicted        

#Defining Inputs and Outputs.
X = np.array([[0,0], [0,1], [1,0], [1,1]])
#Y = np.array([0,1,1,0]) #Loops forever. No convergence.
Y = np.array([0,0,0,1])

per = PerceptronLegacyTerms()
value, Ypredicted = per.converge(X,Y)

print("Output: ", Ypredicted)
print("Final weights: ", value)

