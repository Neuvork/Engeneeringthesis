import cupy as cp
from kernels import dot_cuda_paralell, max_pooling_cuda_paralell, convolve_cuda_paralell
class Neural_Network:

  def parse_input(self,given_layers,input_size,num_nets):
    layers,locs,scales = [],[],[]
    input_size = (1,input_size[0],input_size[1])

    for layer in given_layers:
      print(layer)
      print(layer[2])
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
    self.layers = []
    print("GIVEN LAYERE  BEFORE: ", given_layers)
    given_layers,locs,scales = self.parse_input(given_layers,input_size,num_nets)
    print("GIVEN LAYERE AFTER: ", given_layers)
    print("locs: ", locs)
    print("scales: ", scales)
    for layer,loc,scale in zip(given_layers,locs,scales):
      if layer[0] == 'conv':
        self.layers.append(['conv', cp.random.normal(loc = loc, scale = scale, size = layer[1]).astype(cp.float32)])   #layer[0] -> conv ; layer[1] ->[num_nets, out_channel, in_channel, filter_wdth, filter_height]
      if layer[0] == 'linear':
        self.layers.append(['linear', cp.random.normal(loc = loc, scale = scale, size = layer[1]).astype(cp.float32)])

  def move_to_cpu(self):
    for layer in self.layers:
      layer[1] = cp.asnumpy(layer[1])

  def move_to_gpu(self):
    for layer in self.layers:
      layer[1] = cp.array(layer[1])

  def forward(self, state):
    layer_num = 0
    temp = state.copy()
    first_lin = 0
    for layer in self.layers:
      if layer[0]=='conv':
        temp = convolve_cuda_paralell(temp, layer[1])
        temp = max_pooling_cuda_paralell(temp)
        temp = cp.tanh(temp)
      if layer[0]=='linear':
        if first_lin == 0 and False:
          first_lin+=1
          temp = temp.reshape(-1,layer[1].shape[1])
          #temp = brute_dot(temp, layer[1])
          #temp = brute_dot_single_input(temp, layer[1])
        temp = dot_cuda_paralell(temp, layer[1])

        if layer_num < len(self.layers) - 1:
          temp = cp.tanh(temp)
          layer_num += 1
    return cp.argmax(temp, axis = 1)

  def replace_individual(self, i, individual):
    i = int(i)
    for j in range(len(self.layers)):
      self.layers[j][1][i] = individual[j]
      list_memory_clear(individual)
    del individual


  def get_individual(self, i):
    result_individual = []
    i = int(i)
    for layer in self.layers:
      result_individual.append(layer[1][i].copy())
    return result_individual

  
