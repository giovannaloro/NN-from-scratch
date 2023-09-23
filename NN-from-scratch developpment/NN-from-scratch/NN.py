import numpy as np 

def sigmoid(n):
    return 1/(1 + np.exp(-1*n))

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
            
    #def backpropagate(self,error_goal,dataset): #[[input, output]]
     #   for i in range(len(dataset)):
    #calcolare errore medio 1
        #input forward su ogni input nel dataset 
        #calcolare funzione errore quadratico con output 
        #sommare 
        #dividere numero vettori input outpu
    #controllare se errore medio è minore di errore goal se si stop altrimenti vai avanti 2
    #fare back propagation 3
        #per ogni vettore input output :
            #per ogni peso e bias:
                #calcola derivata
                    #calcola funzione errore attuale
                    #calcola funzione errore facendo variare w/b di h 
                    #ottieni la derivata come f(x) - f(x+h)/h
                #calcola il cambiamento peso/bias come  meno derivata per  learning rate 
                #somma cambiamento alla matrice di aggiornamento del layer 
        #somma matrice aggiornamento pesi e bias 
    #torna al punto 1 


                    







                    

                    



        