import cupy as cp 
import numpy as np
from Engeneeringthesis.Logs import Logs 
from Engeneeringthesis.NeuralNetwork import Neural_Network

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
def cuda_memory_clear():
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()   


class CMA_ES():
  def __init__(self,population,sigma,evaluate_func):
    file = open("LOGS.txt",'w')
    file.write("BUM\n")
    file.close()
    self.dimensionality = population.dimensionality
    self.covariance_matrix = cp.diag(cp.ones(self.dimensionality, dtype = cp.float32))
    print("____allocated")
    cuda_memory_clear()
    self.population = population
    self.sigma = sigma
    self.isotropic = cp.zeros(self.dimensionality) #check it
    self.anisotropic = cp.zeros(self.dimensionality) #check it
    self.evaluate_func = evaluate_func
    self.weights = 0 #0 is just placeholder
    self.logs = Logs([('matrix','covariance'),('population','population'),('number','sigma'),
                      ('vector','isotropic'),('vector','anisotropic'),('vector','mean'),
                      ('number','best-score'),['vector','mean diff']])

  def _indicator_function(self, val, alpha):
    print("___indicator_function start ", val, alpha)
    if val < alpha * self.dimensionality and val > 0:
      print("___indicator_function stop ", 1)
      return 1
    else:
      print("___indicator_function stop ", 0)
      return 0
    return 0


  def update_mean(self, scores,sorted_indices,population,mu):
    print("___update_mean start")
    interesting_values = sorted_indices[:mu]
    valuable_individuals = cp.array(self.population.return_chosen_ones(interesting_values))
    updated_mean = np.sum(valuable_individuals * self.weights.reshape(-1,1),axis = 0)
    print("___update_mean stop", updated_mean)
    return updated_mean

  def update_isotropic(self,mean_act,mean_prev,c_sigma,mu_w):
    print("__update isotropic start, self.isotropic.shape = ", self.isotropic.shape)
    first_term = (1-c_sigma)*self.isotropic
    inversed_covariance_matrix = cp.linalg.cholesky(cp.linalg.inv(self.covariance_matrix))
    test = inversed_covariance_matrix.dot(inversed_covariance_matrix)
    inv_test = cp.linalg.inv(self.covariance_matrix)
    second_term = cp.sqrt(1-((1-c_sigma)**2))*cp.sqrt(mu_w)
    third_term = (cp.array(mean_act)-cp.array(mean_prev))/cp.array(self.sigma)
    ret_val = first_term + second_term*inversed_covariance_matrix.dot(third_term)
    file = open("LOGS.txt", "a")
    file.write("\n update_isotropic min" 
              + str(ret_val.min()) 
              + " mean: " 
              + str(ret_val.mean())
              + " max: "
              + str(ret_val.max()))
    file.close()
    print("__update isotropic stop, self.isotropic.shape = ", self.isotropic.shape)
    return ret_val
  
  def compute_cs(self, alpha, c_1, c_covariance):
    print("__compute_cs start")
    print("__compute_cs end")
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
    print("__update_anisotropic start")
    ret_val = (1 - c_covariance) * self.anisotropic
    ret_val2 = self._indicator_function(self.norm(self.isotropic), alpha)
    ret_val2 *= np.sqrt(1 - (1 - c_covariance ** 2))
    ret_val2 *= np.sqrt(mu_w)
    ret_val3 = (mean_act - mean_prev) / self.sigma
    true_ret_val = ret_val + ret_val2 * ret_val3
    file = open("LOGS.txt", "a")
    file.write("\n Update anisotropic: min: " + str(true_ret_val.min())
                + " mean: " + str(true_ret_val.mean())
                + " max: " + str(true_ret_val.max()))
    file.close()
    print("__update_anisotropic stop")
    return true_ret_val
  
  def _sum_for_covariance_matrix_update(self, scores, sorted_indices, mu, mean_prev): #jakas almbda potrzebna chyba
    print("___sum_for_covariance_matrix_update start")
    interesting_values = sorted_indices[:mu]
    valuable_individuals = cp.array(self.population.return_chosen_ones(interesting_values)) 
    ret_sum = .0
    for i in range(mu):
      ret_sum += self.weights[i] * np.dot((valuable_individuals[i] - mean_prev).reshape(-1,1) #result should be matrix!!!
                / self.sigma, ((valuable_individuals[i] - mean_prev).reshape(1,-1) / self.sigma)  )
    print("___sum_for_covariance_matrix_update stop: ")
    return ret_sum


  def update_covariance_matrix(self, c_1, c_mu, c_s, scores, sorted_indices, mu, mean_prev):
    print("__update_covariance_matrix start")
    discount_factor = 1 - c_1 - c_mu + c_s
    C1 = discount_factor * self.covariance_matrix
    C2 = c_1 * (self.anisotropic.reshape(-1,1).dot(self.anisotropic.reshape(1,-1)))
    C25 = self._sum_for_covariance_matrix_update(scores, sorted_indices, mu, mean_prev)
    C3 = c_mu * self._sum_for_covariance_matrix_update(scores, sorted_indices, mu, mean_prev)
    print("__shapeOfALL",C1.shape,C2.shape,C3.shape,C25.shape)
    print("__update_covariance_matrix stop")
    return C1 + C2 + C3

  def norm(self,vector):
    return cp.sqrt(cp.sum(vector*vector))

  def update_sigma(self,c_sigma,d_sigma):
    print("_update_sigma start")
    temp = cp.sqrt(self.dimensionality)*(1-(1/(4*self.dimensionality)) + (1/(21*self.dimensionality**2)))

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
                + " temp2: " + str(temp2))
    file.close()
    print("_update_sigma stop")
    return self.sigma * temp2



     

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
      scores = self.evaluate_func(self.population, data)
      print(cp.max(scores))
      sorted_indices = cp.argsort(-scores)
      mean_prev = mean_act.copy()
      self.population.parse_to_vector()
      print("___bedzie udpate mean")
      mean_act = self.update_mean(scores,sorted_indices,self.population,mu) #we need to be vectorized here
      print("___bedzie logs log")
      self.logs.log([self.covariance_matrix,self.population.matrix,self.sigma,self.isotropic,self.anisotropic,mean_prev,cp.max(scores),mean_act-mean_prev])
      self.logs.plot()
      self.isotropic = self.update_isotropic(mean_act,mean_prev,c_sigma,mu_w)
      c_s = self.compute_cs(alpha,c_1,c_covariance)
      self.anisotropic = self.update_anisotropic(mean_act,mean_prev,mu_w,c_covariance,alpha)
      self.covariance_matrix = self.update_covariance_matrix(c_1,c_mu,c_s,scores,sorted_indices,mu,mean_prev)
      self.sigma = self.update_sigma(c_sigma,d_sigma)
      self.population.sample(self.covariance_matrix, self.sigma, mean_act, lam)
      self.population.parse_from_vectors()
      file = open("LOGS.txt", "a")
      file.write("\n\n")
      file.close()
    return self.population