from http.client import LENGTH_REQUIRED
from re import I, X
import numpy as np 
import random



def h_value():
    return 10**-8

def sigmoid(n):
    return 1/(1 + np.exp(-1*n))

def s_mod(v,i,j,k,matrix):
    exval = matrix[i][j][k]
    matrix[i][j][k] = exval + v   
    return matrix

def b_mod(v,i,j,matrix):
    exval = matrix[i][j]
    matrix[i][j] = exval + v
    return matrix 

def copy_matrix_array(input_matrix_array, zero=False):
    output_matrix_array = []
    for x in range(len(input_matrix_array)):
        matrix = []
        output_matrix_array.append(matrix)
    for x in range(len(input_matrix_array)):
        for y in range(len(input_matrix_array[x])):
            row = []
            output_matrix_array[x].append(row)
    for x in range(len(input_matrix_array)):
        for y in range(len(input_matrix_array[x])): 
            for z in range(len(input_matrix_array[x][y])):
                output_matrix_array[x][y].append(0)
    return output_matrix_array

def copy_matrix_biases(input_biases_array, zero=False):
    output_biases_array = []
    for x in range(len(input_biases_array)):
        row = []
        output_biases_array.append(row)
    for x in range(len(input_biases_array)):
        for y in range(len(input_biases_array[x])):
            output_biases_array[x].append(0)
    return output_biases_array
    

def print_matrix(matrix):
        for q in range(len(matrix)):
            for w in range(len(matrix[q])):
                    print(matrix[q][w],end="")
            print(" ")
        print("\n")

    

class NN :
    def __init__(self,layers_distribution):
        self.layers_distribution = layers_distribution
        self.weights = []
        self.biases = []
        for x in range(len(self.layers_distribution)-1):
            matrix = []
            for y in range(self.layers_distribution[x]):
                row = []
                for z in range(self.layers_distribution[x+1]):
                    row.append(np.random.rand(1)[0])
                matrix.append(row)
            self.weights.append(matrix)
        for x in range(1, len(self.layers_distribution)):
            biases_layer = []
            for y in range(self.layers_distribution[x]):
                biases_layer.append(np.random.rand(1)[0])
            self.biases.append(biases_layer)

    def update_layers(self, update_layers):
        for q in range(len(self.weights)):
            for w in range(len(self.weights[q])):
                for e in range(len(self.weights[q][w])):
                    self.weights[q][w][e] += update_layers[q][w][e] 

    def update_biases(self, update_biases):
        for q in range(len(self.biases)):
            for w in range(len(self.biases[q])):
                self.biases[q][w] += update_biases[q][w]

    def print_layers(self):
        for q in range(len(self.weights)):
            for w in range(len(self.weights[q])):
                for e in range(len(self.weights[q][w])):
                    print(self.weights[q][w][e],end="")
                print(" ")
            print("\n")

    def print_biases(self):
        for x in range(len(self.biases)):
            for y in range(len(self.biases[x])):
                print(self.biases[x][y], end = "")
                print(" ", end= "")
            print("")        

    def w_mod(self,v,i,j,k):
        exval = self.weights[i][j][k]
        self.weights[i][j][k] += v   
        return exval
    
    def b_mod(self,v,i,j):
        exval = self.biases[i][j]
        self.biases[i][j] += v
        return exval 

    def compute_forward(self,input):
        inner_input = input 
        for i in range(len(self.weights)):
            output = []
            for j in range(len(self.weights[i][0])):
                element = 0
                for k in range(len(self.weights[i])):
                    element += inner_input[k]*self.weights[i][k][j]
                output.append(sigmoid(element - self.biases[i][j]))
            inner_input = output 
        return output
     
    def error(self,input,output):
        error_root = np.subtract(self.compute_forward(input), output)
        return np.multiply(error_root, error_root)
    
    def derivate(self,x,y,z,n,adjustment_matrix,input,output):
        h = 10**-8
        learning_rate = 1
        actual_error = self.error(input,output)
        self.w_mod(h,x,y,z)
        after_modify_error = self.error(input, output)
        self.w_mod(-h,x,y,z)
        derivate = (after_modify_error[n]-actual_error[n])/h
        print("derivate weight")
        print(derivate)
        return s_mod(-derivate*learning_rate,x,y,z,adjustment_matrix)
    
    def derivate_b(self,x,y,n,adjustment_biases,input,output):
        h = 10**-8
        learning_rate = 1
        actual_error = self.error(input,output)
        self.b_mod(h,x,y)
        after_modify_error = self.error(input, output)
        self.b_mod(-h,x,y)
        derivate = (actual_error[n]-after_modify_error[n])/h
        print("derivate bias")
        print(derivate)
        return b_mod(-derivate*learning_rate,x,y,adjustment_biases)

        
    def backpropagate(self,dataset_): #[[input, output]]
        dataset = dataset_
        random.shuffle(dataset)
        dataset = [dataset[i:i + 10] for i in range(0, len(dataset), 10)]
        original_weights = self.weights
        original_biases = self.biases
        adjustment_biases = copy_matrix_biases(self.biases)
        adjustment_weights = copy_matrix_array(self.weights)
        for e in range(len(dataset)):
            for t in range(len(dataset[e])):
                for n in range(len(self.weights[len(self.weights)-1][0])):
                    for x in range(len(self.weights)):
                        for y in range(len(self.weights[x])):
                            for z in range(len(self.weights[x][y])):
                                self.derivate(x,y,z,n,adjustment_weights,dataset[e][t][0],dataset[e][t][1])
            for t in range(len(dataset[e])):
                    for n in range(len(self.weights[len(self.weights)-1][0])):
                        for x in range(len(self.biases)):
                            for y in range(len(self.biases[x])):
                                self.derivate_b(x,y,n,adjustment_biases,dataset[e][t][0],dataset[e][t][1])
            ex = self.weights
            self.weights = adjustment_weights
            self.print_layers()
            self.weights = ex
            self.update_layers(adjustment_weights)
            self.update_biases(adjustment_biases)
            adjustment_biases = copy_matrix_biases(self.biases)
            adjustment_weights = copy_matrix_array(self.weights)
 


