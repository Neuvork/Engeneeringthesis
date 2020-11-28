import cupy as cp
import numpy as np
no_debug = 1
basic_debug_mode = 2
super_debug_mode = 3
only_interesting = 5
DEBUG_MODE = only_interesting
from Engeneeringthesis.kernels import dot_cuda_paralell, max_pooling_cuda_paralell, convolve_cuda_paralell
class Neural_Network:


  def __init__(self,num_nets,input_size,given_layers,loc=0,scale=1, cage_dimensionalities = None):#after init in neuronized state
    self.mempool = cp.get_default_memory_pool()
    self.pinned_mempool = cp.get_default_memory_pool()
    self.population_size = num_nets
    self.input_size = input_size
    self.input_layers = given_layers
    self.vectorized = False #if NN is in state of being vectorized or neuronized
    self.layers = [] #empty if in vectorized,neural network if in neuronized
    self.matrix = None #empty if in neuronized, vector if in vectorized
    self.layers_shapes = self.parse_input(given_layers,input_size,num_nets) #remember the shape of network,and parse user input
    self.dimensionality = self.compute_dimensionality()
    self.cage_dimensionalities = cage_dimensionalities
    for layer in self.layers_shapes:
      if layer[0] == 'conv':
        self.layers.append(['conv', cp.random.normal(loc = loc, scale = scale, size = layer[1]).astype(cp.float32)])   #layer[0] -> conv ; layer[1] ->[num_nets, out_channel, in_channel, filter_wdth, filter_height]
      if layer[0] == 'linear':

        self.layers.append(['linear', cp.random.normal(loc = loc, scale = scale, size = layer[1]).astype(cp.float32)])  
  

  def cuda_memory_clear(self):
    self.mempool.free_all_blocks()
    self.pinned_mempool.free_all_blocks()          

  def parse_to_vector(self): # every individual is getting trapnsfered to vector
    ret_mat = np.zeros((self.population_size, self.dimensionality),dtype = np.float32)
    index = 0
    for layer in self.layers:
      i = 0
      for individual in layer[1]:
        ret_mat[i][index:] = cp.asnumpy(individual.flatten())
        i += 1 
      index += layer[1][0].flatten().size
    self.layers = []
    self.cuda_memory_clear()
    self.matrix = cp.array(ret_mat, dtype = cp.float32)
    self.vectorized = True

  def list_memory_clear(self, lista):
    for i in range(len(lista)):
      del lista[0]

  def parse_input(self,given_layers,input_size,num_nets):
    layers = []
    input_size = (input_size[0],input_size[1],input_size[2])

    for layer in given_layers:

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

    return layers
  
  def compute_dimensionality(self):
    number_of_weights = 0
    for layer_shape in self.layers_shapes:
      weights_in_layer = 1
      for number in layer_shape[1][1:]:
        weights_in_layer *= number
      number_of_weights += weights_in_layer
    return number_of_weights



  def sample(self,B,D, sigma, mean, lam):
    self.layers = [] #cleaning previous population
    self.cuda_memory_clear()
    #concat sampled vectors and parse them
    ret_mat = cp.zeros((lam, self.dimensionality),dtype = cp.float32)
    #L = cp.linalg.cholesky(covariance_matrix*(sigma**2)).astype(cp.float32)
    for i in range(lam):
      ret_mat[i] = self.multivariate_cholesky(mean,B,D,sigma)
      #ret_mat[i] = cp.random.multivariate_normal(mean, covariance_matrix * (sigma**2))
      self.cuda_memory_clear()
    self.matrix = ret_mat
    self.vectorized = True

  def multivariate_cholesky(self,mean,B,D,sigma):
    vector = cp.random.normal(loc = 0,scale = 1,size = self.dimensionality,dtype = cp.float32)
    ret_val = sigma*B.dot(D*vector) + mean
    return ret_val

  def caged_sample(self,covariance_matrices, sigmas, means, lam):
    self.layers = [] #cleaning previous population
    self.cuda_memory_clear()
    #concat sampled vectors and parse them
    ret_mat = cp.zeros((lam, self.dimensionality),dtype = cp.float32)
    L = []
    for i in range(len(self.cage_dimensionalities)):
      L.append(cp.linalg.cholesky(covariance_matrices[i]*(sigmas[i]**2)).astype(cp.float32))
    for i in range(lam):
      ret_mat[i] = self.caged_multivariate_cholesky(means,L)
    self.cuda_memory_clear()
    self.matrix = ret_mat
    self.vectorized = True
  
  def caged_multivariate_cholesky(self, means, cholesky_covariances):
    vector = cp.array([])
    for i in range(len(means)):
      sampled_vector = cp.random.normal(loc = 0,scale = 1,size = cholesky_covariances[i].shape[0],dtype = cp.float32)
      vector = cp.concatenate((vector, cholesky_covariances[i].dot(sampled_vector) + means[i]))
    return vector

  def mult(self, l):
    ret_val = 1
    for number in l:
      ret_val *= number
    return ret_val

  def parse_from_vectors(self):
    numbers = []
    self.matrix = cp.asnumpy(self.matrix)
    self.cuda_memory_clear()
    for layer in self.layers_shapes:
      print(layer[1])
      numbers.append(self.mult(layer[1][1:]))
    start = 0
    it = 0
    for number in numbers:
      self.layers.append((self.layers_shapes[it][0],cp.array(self.matrix[:,start:(start+number)]).reshape(self.layers_shapes[it][1])))
      it+=1
    self.matrix = None
    self.vectorized = False

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
      self.list_memory_clear(individual)
    del individual

#self.cage dimensionalities 
  def return_chosen_ones(self, indices, number_of_cage = None):
    if not self.vectorized:
        self.parse_to_vector()
    
    if number_of_cage == None:
      return self.matrix[indices]
    else:
      begin = self.cage_dimensionalities[:number_of_cage].sum()
      move = self.cage_dimensionalities[number_of_cage]
      return self.matrix[indices, begin : begin + move]
      
      




  def get_individual(self, i):
    result_individual = []
    i = int(i)
    for layer in self.layers:
      result_individual.append(layer[1][i].copy())
    return result_individual
