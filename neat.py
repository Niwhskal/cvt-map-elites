import numpy as np


class NN():
    def __init__(self, in_shape, out_shape, n_mid_neurons):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.n_mid_neurons = n_mid_neurons
        self.node_genes = {}
        self.connect_genes = {}

    def create_node_list(self):
        count = 0
        for i in range(in_shape):
            self.node_genes[count] = "sensor"
            count += 1

        for j in range(out_shape):
            self.node_genes[count] = "output"
            count += 1

        for k in range(mid_neurons):
            self.node_genes[count] = "hidden"
            count += 1

    def create_connection_list(self):


