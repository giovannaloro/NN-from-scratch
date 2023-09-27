import numpy as np 
import NN 
my_nn = NN.NN([2,4,3])
my_nn.print_layers()
my_nn.print_biases()
out = my_nn.compute_forward([1,2])
print("pretraining")
print(out)
dataset = [
    [[1, 2], [1, 3, 3]],
    [[3, 4], [3, 7, 7]],
    [[5, 6], [5, 11, 11]],
    [[7, 8], [7, 15, 15]],
    [[9, 10], [9, 19, 19]],
    ]
my_nn.backpropagate(8,dataset)
print("aftertraining")
out = my_nn.compute_forward([4,2])
print(out)
