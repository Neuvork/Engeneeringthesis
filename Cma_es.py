import cupy as cp 
import numpy as np
from Engeneeringthesis.Logs import Logs 
from Engeneeringthesis.NeuralNetwork import Neural_Network
from scipy.linalg import sqrtm

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
def cuda_memory_clear():
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()   


class CMA_ES():
  def __init__(self,population,sigma,evaluate_func, logs, dimensionality = None, number_of_cage = None):
    file = open("LOGS.txt",'w')
    file.write("BUM\n")
    file.close()
    self._loops_number = 0
    self.hp_loops_number = 3
    self.dimensionality = None
    if dimensionality == None:
      self.dimensionality = population.dimensionality
    else:
      self.dimensionality = dimensionality
    self.number_of_cage = number_of_cage
    self.B_matrix = cp.diag(cp.ones(self.dimensionality,dtype = cp.float32))
    self.D_matrix = cp.ones(self.dimensionality,dtype = cp.float32).reshape(-1,1).flatten()
    print(self.B_matrix.shape,self.D_matrix.shape)
    self.covariance_matrix = (self.B_matrix.dot(cp.diag(self.D_matrix**2))).dot(self.B_matrix.T)
    #self.covariance_matrix = cp.diag(cp.ones(self.dimensionality, dtype = cp.float32))
    self.invert_sqrt_covariance_matrix = (self.B_matrix.dot(cp.diag(self.D_matrix**-1))).dot(self.B_matrix.T)
    cuda_memory_clear()
    self.population = population
    self.sigma = sigma
    self.isotropic = cp.zeros(self.dimensionality, dtype = cp.float32) #check it
    self.anisotropic = cp.zeros(self.dimensionality, dtype = cp.float32) #check it
    self.evaluate_func = evaluate_func
    self.weights = 0 #0 is just placeholder
    self.logs = logs

  def _indicator_function(self, val, alpha):
    if val < alpha * self.dimensionality and val > 0:
      return 1
    else:
      return 0
    return 0


  def update_mean(self, scores,sorted_indices,mu):
    interesting_values = sorted_indices[:mu]
    valuable_individuals = cp.array(self.population.return_chosen_ones(interesting_values, self.number_of_cage))
    updated_mean = np.sum(valuable_individuals * self.weights.reshape(-1,1),axis = 0)
    file = open("LOGS.txt", "a")
    file.write("number_of_cage: " + str(self.number_of_cage) +" valueable_individuals: " + str(valuable_individuals))
    return updated_mean

  def update_isotropic(self,mean_act,mean_prev,c_sigma,mu_w):
    file = open("LOGS.txt", "a")

    first_term = (1-c_sigma)*self.isotropic

    #inversed_covariance_matrix = cp.linalg.cholesky(cp.linalg.inv(self.covariance_matrix)).astype(cp.float32)
    #inversed_covariance_matrix = cp.array(sqrtm(cp.asnumpy(cp.linalg.inv(self.covariance_matrix))), dtype = cp.float32)
    second_term = (cp.sqrt(1-((1-c_sigma)**2))*cp.sqrt(mu_w)).astype(cp.float32)
    third_term = (cp.array(mean_act, dtype = cp.float32)-cp.array(mean_prev, dtype=cp.float32))/cp.array(self.sigma, dtype=cp.float32)
    ret_val = first_term + second_term*self.invert_sqrt_covariance_matrix.dot(third_term)

    
    file.write("\n \n update_isotropic second_term: \n first_part:  " 
              + str(ret_val[0].dtype) 
              + ", mean_act: "
              + str(mean_act)
              + "\n mean_prev: "
              + str(mean_prev)
              + "\n"
              + str(self.sigma)
              + "\n sigma: "
              + str(ret_val)
              + "\n\n"
              )
    file.close()
    self.isotropic = ret_val
  
  def compute_cs(self, alpha, c_1, c_covariance):
    ret_val = (1 - self._indicator_function(cp.sqrt(cp.sum(self.isotropic ** 2)), alpha)) * c_1 * c_covariance * (2 - c_covariance)
    file = open("LOGS.txt", "a")
    file.write("\n compute_cs: min: " 
              + str(ret_val.min())
              + " mean: "
              + str(ret_val.mean())
              + " max: "
              + str(ret_val.max()))
    file.close()
    return ret_val

  def update_anisotropic(self, mean_act,mean_prev,mu_w,c_covariance,alpha):
    ret_val = (1 - c_covariance) * self.anisotropic
    ret_val2 = self._indicator_function(self.norm(self.isotropic), alpha)
    ret_val2 *= np.sqrt(1 - (1 - c_covariance ** 2))
    ret_val2 *= np.sqrt(mu_w)
    ret_val3 = (mean_act - mean_prev) / self.sigma
    true_ret_val = ret_val + ret_val2 * ret_val3
    file = open("LOGS.txt", "a")
    file.write("\n Update anisotropic: "
                + " mean: " + str(ret_val.mean())
                + " mean2: " + str(ret_val2.mean())
                + " mean3: " + str(ret_val3.mean())
                )
    file.close()
    self.anisotropic = true_ret_val
  
  def _sum_for_covariance_matrix_update(self, scores, sorted_indices, mu, mean_prev): #jakas almbda potrzebna chyba
    interesting_values = sorted_indices[:mu]
    valuable_individuals = cp.array(self.population.return_chosen_ones(interesting_values, self.number_of_cage), cp.float32) 
    ret_sum = .0
    for i in range(mu):
      ret_sum += self.weights[i] * np.dot((valuable_individuals[i] - mean_prev).reshape(-1,1) #result should be matrix!!!
                / self.sigma, ((valuable_individuals[i] - mean_prev).reshape(1,-1) / self.sigma)  )
    return ret_sum


  def update_covariance_matrix(self, c_1, c_mu, c_s, scores, sorted_indices, mu, mean_prev):
    file = open("LOGS.txt", "a")
    file.write( " Przed pajacowaniem: dtype: "
                + str(self.covariance_matrix.dtype)
                )
    discount_factor = 1 - c_1 - c_mu + c_s
    C1 = discount_factor * self.covariance_matrix
    C2 = (c_1 * (self.anisotropic.reshape(-1,1).dot(self.anisotropic.reshape(1,-1)))).astype(cp.float32)
    C3 = (c_mu * self._sum_for_covariance_matrix_update(scores, sorted_indices, mu, mean_prev)).astype(cp.float32)
    

    self.covariance_matrix = C1 + C2 + C3
    if self._loops_number == self.hp_loops_number:
      self.covariance_matrix = cp.triu(self.covariance_matrix) + cp.triu(self.covariance_matrix,1)
      self._loops_number = 0
      self.D_matrix,self.B_matrix = cp.linalg.eigh(self.covariance_matrix)
      self.D_matrix = cp.sqrt(self.D_matrix)
      self.invert_sqrt_covariance_matrix = (self.B_matrix.dot(cp.diag(self.D_matrix**-1))).dot(self.B_matrix.T)
    file.write( " Po pajacowaniem: dtype: "
                + str(self.covariance_matrix.dtype)
                + " C1: " + str(C1.dtype) + ", "
                + " C2: " + str(C2.dtype) + ", "
                + " C3: " + str(C3.dtype) 
                )
    file.close()


  def norm(self,vector):
    return cp.sqrt(cp.sum(vector*vector))

  def update_sigma(self,c_sigma,d_sigma):
    temp = cp.sqrt(self.dimensionality, dtype = cp.float32)*(1-(1/(4*self.dimensionality)) + (1/(21*self.dimensionality**2)))

    temp2 = cp.exp((c_sigma/d_sigma)*((self.norm(self.isotropic)/temp)-1))
    ret_val = self.sigma * temp2
    file = open("LOGS.txt", "a")
    file.write("\n update_sigma: min: " + str(ret_val.min())
                + " mean: " + str(ret_val.mean())
                + " max: "  + str(ret_val.max())
                + " dimensionality: " + str(self.dimensionality)
                + " temp: " + str(temp)
                + " temp2 part one: " + str(cp.exp((c_sigma/d_sigma)))
                + " temp2 part two: " + str(((self.norm(self.isotropic)/temp)-1))
                + " temp2: " + str(temp2)
                + " self.sigma: " + str(self.sigma)
                + " self.new_sigma: " + str(self.sigma * temp2)
                + " cage_number: " + str(self.number_of_cage))
    file.close()
    self.sigma*=temp2



     

  def fit(self, data, mu, lam, iterations): # mu is how many best samples from population, lam is how much we generate
    mean_act = cp.zeros(self.dimensionality)
    #constant
    self.weights = cp.log(mu+1/2) - cp.log(cp.arange(1,mu+1))
    self.weights = self.weights/cp.sum(self.weights)
    mu_w = 1/cp.sum(self.weights**2)
    
    dimension = 7840
    #c_sigma = (mu_w + 2)/(dimension + mu_w + 5)
    #d_sigma = 1 + 2*max([0,cp.sqrt((mu_w - 1)/(dimension + 1)) - 1]) + c_sigma #dampening parameter could probably be hyperparameter, wiki says it is close to 1 so whatever
    #c_covariance = (4 + mu_w/dimension)/(dimension + 4 + 2*mu_w/dimension) # c_covariance * 100 not working
    #c_1 = 2/(dimension**2)
    #c_mu = min([1-c_1,2*(mu_w - 2 + 1/mu_w)/(((dimension+2)**2)+mu_w)])

    c_1 = 2/(self.dimensionality**2)
    c_sigma = (mu_w + 2)/(self.dimensionality + mu_w + 5)
    d_sigma = 1 + 2*max([0,cp.sqrt((mu_w - 1)/(self.dimensionality + 1)) - 1]) + c_sigma #dampening parameter could probably be hyperparameter, wiki says it is close to 1 so whatever
    c_covariance = (4 + mu_w/self.dimensionality)/(self.dimensionality + 4 + 2*mu_w/self.dimensionality) # c_covariance * 100 not working
    c_mu = min([1-c_1,2*(mu_w - 2 + 1/mu_w)/(((self.dimensionality+2)**2)+mu_w)])
    
    file = open("PARAMS.txt", "w")
    file.write("c_1: " + str(c_1) + "\n")
    file.write("c_mu: " + str(c_mu) + "\n")
    file.write("c_sigma: " + str(c_sigma) + "\n")
    file.write("d_sigma: " + str(d_sigma) + "\n")
    file.write("c_covariance: " + str(c_covariance) + "\n")
    file.close()
    
    alpha = 1.5
    #body 
    for i in range(iterations):
      self._loops_number += 1
      scores = self.evaluate_func(self.population, data)
      print(cp.max(scores))
      sorted_indices = cp.argsort(-scores)
      mean_prev = mean_act.copy()
      self.population.parse_to_vector()
      print("___bedzie udpate mean")
      mean_act = self.update_mean(scores,sorted_indices,mu) #we need to be vectorized here
      print("___bedzie logs log")
      self.logs.log([self.covariance_matrix,self.population.matrix,self.sigma,self.isotropic,self.anisotropic,mean_prev,cp.max(scores),mean_act-mean_prev])
      self.logs.plot()
      self.update_isotropic(mean_act,mean_prev,c_sigma,mu_w)
      c_s = self.compute_cs(alpha,c_1,c_covariance)
      self.update_anisotropic(mean_act,mean_prev,mu_w,c_covariance,alpha)
      self.update_covariance_matrix(c_1,c_mu,c_s,scores,sorted_indices,mu,mean_prev)
      self.update_sigma(c_sigma,d_sigma)
      self.population.sample(self.B_matrix, self.D_matrix, self.sigma, mean_act, lam)
      self.population.parse_from_vectors()
      file = open("LOGS.txt", "a")
      file.write("\n\n")
      file.close()
    return self.population