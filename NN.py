from http.client import LENGTH_REQUIRED
from re import I, X
import numpy as np 

def h_value():
    return 10**-8

def sigmoid(n):
    return 1/(1 + np.exp(-1*n))

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
                print(x,y,z)
                output_matrix_array[x][y].append(0)
    return output_matrix_array

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
                    row.append(np.random.rand(1))
                matrix.append(row)
            self.weights.append(matrix)
        for x in range(1, len(self.layers_distribution)):
            biases_layer = []
            for y in range(self.layers_distribution[x]):
                biases_layer.append(np.random.rand(1))
            self.biases.append(biases_layer)


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
        self.weights[i][j][k] = v   
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
     
    def error(self,input ,output):
        error_root = np.subtract(NN.compute_forward(input), output)
        return np.multiply(error_root, error_root)
    
    """
      
           
    def backpropagate(self,error_goal,dataset): #[[input, output]]
        original_weights = self.weights
        original_biases = self.biases
        adjustment_
        for e in range(len(dataset)):
            for n in range(len(self.weights[len(self.weights)][0])):
                for x in range(len(self.weights)):
                    for y in range(len(self.weights[x])):
                        for z in range(len(self.weights[x][y])):

            


        #per ogni vettore input output :
            #per ogni neurone 
                #per ogni peso e bias:
                    #calcola derivata
                        #calcola funzione errore attuale
                        #calcola funzione errore facendo variare w/b di h 
                        #ottieni la derivata come f(x) - f(x+h)/h
                    #calcola il cambiamento peso/bias come  meno derivata per  learning rate 
                    #somma cambiamento alla matrice di aggiornamento del layer 
            #somma matrice aggiornamento pesi e bias 

"""
