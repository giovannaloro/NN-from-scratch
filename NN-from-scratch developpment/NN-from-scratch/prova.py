import NN 
my_nn = NN.NN([2,3,2])
my_nn.print_layers()
my_nn.print_biases()
out = my_nn.compute_forward([1,2])
print(out)
