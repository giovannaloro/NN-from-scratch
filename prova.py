import numpy as np 
import NN 
import math
import sys

dataset = []
for x in range(1, 200):
    y = x 
    dataset.append([[x, y], [1]])
for x in range(1, 200):
    dataset.append([[x, -x], [0]])

my_nn = NN.NN([2,5,3,8,1])
#my_nn.print_layers()
print ( " " )
#my_nn.print_biases()
print("pretraining")
out = my_nn.compute_forward([1,1])
print(out)
my_nn.backpropagate(dataset)
my_nn.backpropagate(dataset)
my_nn.backpropagate(dataset)
my_nn.backpropagate(dataset)
print("aftertraining")
print("should give 1")
out = my_nn.compute_forward([1,1])
print(out)
print("should give zero")
out = (my_nn.compute_forward([8,-8]))
print(out)
while(True):
    x = int(input("x"))
    if (x == "no"):
        sys.exit()
    y = int(input("y"))
    print(my_nn.compute_forward([x,y]))
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