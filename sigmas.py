import cupy as cp
import numpy as np
no_debug = 1
basic_debug_mode = 2
super_debug_mode = 3
only_interesting = 5
DEBUG_MODE = no_debug

class Sigmas_Neural_Network:
    def parse_input(self,given_layers,input_size,num_nets):
      layers,locs,scales = [],[],[]
      input_size = (input_size[0],input_size[1],input_size[2])

      for layer in given_layers:
        locs.append(layer[2][0])
        scales.append(layer[2][1])

        if layer[0] == 'conv':
          layers.append((layer[0],[num_nets,layer[1][0],input_size[0],layer[1][1],layer[1][2]]))
          input_size = (layer[1][0],input_size[1]-layer[1][1]+1,input_size[2]-layer[1][2]+1)
          input_size = (input_size[0],np.ceil(input_size[1]/2),np.ceil(input_size[2]/2))

        if layer[0] == 'linear':
          temp = 1
          for i in input_size:
            temp *= i
          input_size = int(temp)
          layers.append((layer[0],[num_nets,input_size,layer[1]]))
          input_size = layer[1]

      return layers,locs,scales


    def __init__(self,num_nets,input_size,given_layers):
        self.sigmas_size = num_nets
        self.input_size = input_size
        self.input_layers = given_layers
        self.layers_sigmas = []
        given_layers,locs,scales = self.parse_input(given_layers,input_size,num_nets)
        for layer,loc,scale in zip(given_layers,locs,scales):
            if layer[0] == 'conv':
                self.layers_sigmas.append(['conv', cp.abs(cp.random.normal(loc = loc, scale = scale, size = (layer[1]))).astype(cp.float32)])   #layer[0] -> conv ; layer[1] ->[num_nets, out_channel, in_channel, filter_wdth, filter_height]
            if layer[0] == 'linear':
                self.layers_sigmas.append(['linear', cp.abs(cp.random.normal(loc = loc, scale = scale, size = (layer[1]))).astype(cp.float32)])

    def list_memory_clear(self, lista):
      for i in range(len(lista)):
        del lista[0]

    def replace_individual(self, i, individual):
        i = int(i)
        for j in range(len(self.layers_sigmas)):
            self.layers_sigmas[j][1][i] = individual[j]
        self.list_memory_clear(individual)
        del individual

    def get_individual(self, i):
        result_individual = []
        i = int(i)
        for layer in self.layers_sigmas:
            result_individual.append(layer[1][i].copy())
        return result_individual

    def mutate(self, mutation_parameter_individual, mutation_parameter_coordinate):
        if DEBUG_MODE % basic_debug_mode == 0:
          print("SIGMAS MUTATION START", self.layers_sigmas[0][1].shape)
        flag = 0
        random_individual_mutation = cp.random.normal(loc = 0., scale = mutation_parameter_individual, size = (self.layers_sigmas[0][1].shape[0], 1, 1, 1, 1))
        for layer in self.layers_sigmas:
            if layer[0]=='linear' and flag==0:
                flag=1
                random_individual_mutation = random_individual_mutation.reshape(self.layers_sigmas[0][1].shape[0], 1 ,1)
            random_weight_mutation = cp.random.normal(loc = 0., scale = mutation_parameter_coordinate, size = layer[1].shape)
            layer[1] *= cp.exp(random_weight_mutation + random_individual_mutation).astype(cp.float32)
        if DEBUG_MODE % basic_debug_mode == 0:
          print("SIGMAS MUTATION STOP", self.layers_sigmas[0][1].shape)
