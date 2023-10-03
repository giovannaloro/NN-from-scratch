# NN-from-scratch
An basic library to create and train small neural network
#Use
* Import NN
* Create a new network with NN.network() --> network 
* Train a network with network.train()
* Calculate a network prediction with network.predict()
#Example
dataset = [[[1,-1],[0]],[[1,1],[1]]]
network = NN.Network([2,5,3,8,1],-10,10)
network.train(dataset,1,0.7)
network.predict([1,-1])
