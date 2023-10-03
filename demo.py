import NN 

dataset = []
for x in range(1, 200):
    y = x 
    dataset.append([[x, y], [1]])
for x in range(1, 200):
    dataset.append([[x, -x], [0]])
network = NN.Network([2,5,3,8,1],-10,10)
network.train(dataset,1,0.7)
print("should be 0: ", network.predict([30,-30]))
print("should be 1:", network.predict([111,111]))
x = int(input("x"))
y = int(input("y"))
print(network.predict([x,y]))

