import numpy as np 
import NN 
import math

dataset = []
for x in range(1, 1010):
    y = x * 2
    dataset.append([[x, y], [x,x+y]])

my_nn = NN.NN([2,4,4,16,2])
my_nn.print_layers()
print ( " " )
my_nn.print_biases()
print("pretraining")
out = my_nn.compute_forward([1,1])
print(out)
my_nn.backpropagate(dataset)
print("aftertraining")
out = my_nn.compute_forward([1,1])
print(out)
"""
def ln(x,y):
    return math.log(x**2+y,2.71828)

def lnx(x,y):
    return  ((2*x)/(x**2+y))

def approsimate_lnx(x,y):
    h = 10**-8
    return (ln(x+h,y)-ln(x,y))/h

print(ln(2,3))
print(lnx(2,3))
print(approsimate_lnx(2,3))
"""