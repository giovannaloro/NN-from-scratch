import numpy


class NN :
    def __init__(self,layer_distribution):
        self.layer_distribution = layer_distribution
        self.layers = []
        for i in  len(self.layer_distribution):
            matrix = []
            for j in self.layer_distribution[i]:
                row = []
                for k in self.layer_distribution[i+1]:
                    row.append(numpy.random.rand())
                matrix.append(row)
        self.layers.append(matrix)
                    



        