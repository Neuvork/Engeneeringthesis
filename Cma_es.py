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
  def __init__(self,population,sigma,evaluate_func, logs, dimensionality = None, param_dimensionality = None, number_of_cage = None, hp_loops_number = 0, patience = None):
    file = open("LOGS.txt",'w')
    file.close()
    self._loops_number = 1
    self.hp_loops_number = self._loops_number + hp_loops_number
    self.dimensionality = None
    self.param_dimensionality = None
    if dimensionality == None:
      self.dimensionality = population.dimensionality
    else:
      self.dimensionality = dimensionality
    if param_dimensionality == None:
      self.param_dimensionality = self.dimensionality
    else:
      self.param_dimensionality = param_dimensionality

    self.number_of_cage = number_of_cage
    self.B_matrix = cp.diag(cp.ones(self.dimensionality,dtype = cp.float32))
    self.D_matrix = cp.ones(self.dimensionality,dtype = cp.float32).reshape(-1,1).flatten()
    self.covariance_matrix = (self.B_matrix.dot(cp.diag(self.D_matrix**2))).dot(self.B_matrix.T)
    self.invert_sqrt_covariance_matrix = (self.B_matrix.dot(cp.diag(self.D_matrix**-1))).dot(self.B_matrix.T)
    
    cuda_memory_clear()
    self.population = population

    self.sigma = sigma
    self.delta_sigma = 1


    #sigma heurestics
    self.patience = patience
    if self.patience != None:
      self.patience *= self.hp_loops_number
    self.starting_sigma = self.sigma
    self.sigma_drop = 499/500
    self.best_validation = 0
    self.should_heat_up = False
    self.iterations_without_improvment = 0

    self.isotropic = cp.zeros(self.dimensionality, dtype = cp.float32) #check it
    self.d_isotropic = cp.zeros(self.dimensionality, dtype = cp.float32) #check it
    
    self.anisotropic = cp.zeros(self.dimensionality, dtype = cp.float32) #check it
    self.d_anisotropic = cp.zeros(self.dimensionality, dtype = cp.float32) #check it

    self.evaluate_func = evaluate_func
    self.weights = 0 #0 is just placeholder
    self.logs = logs
    file = open("LOGS.txt", "a")
    file.write("covariance_matrix: " + str(self.covariance_matrix.dtype) + "\n"
                + "invert_sqrt_covariance_matrix: " + str(self.invert_sqrt_covariance_matrix.dtype) + "\n")
    file.close()
  def _indicator_function(self, val, alpha):
    if val < alpha * self.param_dimensionality and val > 0:
      return 1
    else:
      return 0
    return 0


  def update_mean(self, scores,sorted_indices,mu):
    interesting_values = sorted_indices[:mu]
    valuable_individuals = cp.array(self.population.return_chosen_ones(interesting_values, self.number_of_cage))
    updated_mean = np.sum(valuable_individuals * self.weights.reshape(-1,1),axis = 0,dtype=np.float32)
    file = open("LOGS.txt", "a")
    file.write("number_of_cage: " + str(self.number_of_cage) +" valueable_individuals: " + str(valuable_individuals))
    file.close()
    return updated_mean

  def update_isotropic(self,mean_act,mean_prev,c_sigma,mu_w):
    file = open("LOGS.txt", "a")

    first_term = (1-c_sigma)*self.isotropic.astype(cp.float32)

    second_term = (cp.sqrt(1-((1-c_sigma)**2))*cp.sqrt(mu_w)).astype(cp.float32)
    third_term = (cp.array(mean_act, dtype = cp.float32)-cp.array(mean_prev, dtype=cp.float32))/cp.array(self.sigma, dtype=cp.float32)
    ret_val = first_term + second_term*self.invert_sqrt_covariance_matrix.dot(third_term)


    file.write("\n  " 
                + "first term: " + str(first_term.dtype) + "\n"
                + "second term: " + str(second_term.dtype) + "\n"
                + "third_term: " + str(third_term.dtype) + "\n"
              )
    file.close()
    self.d_isotropic += second_term*self.invert_sqrt_covariance_matrix.dot(third_term)
    if self.hp_loops_number == self._loops_number:
      self.isotropic = (1-c_sigma)*self.isotropic + self.d_isotropic/self.hp_loops_number
      self.isotropic = self.isotropic.astype(cp.float32)
      self.d_isotropic = cp.zeros(self.dimensionality, dtype = cp.float32)
  
  def compute_cs(self, alpha, c_1, c_covariance):
    ret_val = (1 - self._indicator_function(cp.sqrt(cp.sum(self.isotropic ** 2)), alpha)) * c_1 * c_covariance * (2 - c_covariance)
    file = open("LOGS.txt", "a")
    file.write( " ret_val: " + str(ret_val.dtype) + "\n")
    file.close()
    return ret_val.astype(cp.float32)

  def update_anisotropic(self, mean_act,mean_prev,mu_w,c_covariance,alpha):
    ret_val = (1 - c_covariance).astype(cp.float32) * self.anisotropic
    ret_val2 = self._indicator_function(self.norm(self.isotropic), alpha)
    ret_val2 *= np.sqrt(1 - (1 - c_covariance ** 2))
    ret_val2 *= (np.sqrt(mu_w))
    ret_val2 = ret_val2.astype(cp.float32)
    ret_val3 = (mean_act - mean_prev) / cp.float32(self.sigma)
    ret_val3 = ret_val3.astype(cp.float32)
    true_ret_val = ret_val + ret_val2 * ret_val3
    file = open("LOGS.txt", "a")
    file.write("\n Update anisotropic: "
                + "ret_val: " + str(ret_val.dtype) + "\n"
                + "ret_val2: " + str(ret_val2.dtype) + "\n"
                + "ret_val3: " + str(ret_val3.dtype) + "\n"
                + "true ret val: " + str(true_ret_val.dtype) + "\n"
                )
    file.close()

    self.d_anisotropic += ret_val2 + ret_val3
    if self.hp_loops_number == self._loops_number:
      self.anisotropic = (1 - c_covariance)*self.anisotropic + self.d_anisotropic/self.hp_loops_number
      self.d_anisotropic = cp.zeros(self.dimensionality, dtype = cp.float32) 
      self.anisotropic = self.anisotropic.astype(cp.float32)

  
  def _sum_for_covariance_matrix_update(self, scores, sorted_indices, mu, mean_prev): #jakas almbda potrzebna chyba
    interesting_values = sorted_indices[:mu]
    valuable_individuals = cp.array(self.population.return_chosen_ones(interesting_values, self.number_of_cage), cp.float32) 
    ret_sum = .0
    for i in range(mu):
      ret_sum += self.weights[i] * np.dot((valuable_individuals[i] - mean_prev).reshape(-1,1) #result should be matrix!!!
                / self.sigma, ((valuable_individuals[i] - mean_prev).reshape(1,-1) / self.sigma)  )
    file = open("LOGS.txt", "a")
    file.write("\n _sum_for_covariance_matrix_update: "
                + "ret_sum: " + str(ret_sum.dtype) + "\n"
                )
    file.close() 
    return ret_sum.astype(cp.float32)


  def update_covariance_matrix(self, c_1, c_mu, c_s, scores, sorted_indices, mu, mean_prev):
    file = open("LOGS.txt", "a")
    file.write( " Przed pajacowaniem: dtype: "
                + str(self.covariance_matrix.dtype)
                )
    discount_factor = 1 - (c_1 - c_mu + c_s)/self.hp_loops_number
    C1 = discount_factor.astype(cp.float32) * self.covariance_matrix
    C2 = (c_1 * (self.anisotropic.reshape(-1,1).dot(self.anisotropic.reshape(1,-1)))).astype(cp.float32)
    C3 = (c_mu * self._sum_for_covariance_matrix_update(scores, sorted_indices, mu, mean_prev)).astype(cp.float32)


    self.covariance_matrix = C1 + (C2 + C3)/self.hp_loops_number
    self.covariance_matrix = self.covariance_matrix.astype(cp.float32)
    if self._loops_number == self.hp_loops_number:
      self.covariance_matrix = cp.triu(self.covariance_matrix) + cp.triu(self.covariance_matrix,1).T
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
    temp = cp.sqrt(self.param_dimensionality, dtype = cp.float32)*(1-(1/(4*self.param_dimensionality)) + (1/(21*self.param_dimensionality**2)))

    temp2 = cp.exp((c_sigma/d_sigma)*((self.norm(self.isotropic)/temp)-1)).astype(cp.float32)
    ret_val = cp.float32(self.sigma) * temp2
    file = open("LOGS.txt", "a")
    file.write("\n update_sigma: min: " + str(ret_val.min())
                + "temp: " + str(temp.dtype) + "\n"
                + "temp2: " + str(temp2.dtype) + "\n"
                + "ret_val: " + str(ret_val.dtype) + "\n")
    file.close()
    self.delta_sigma *= temp2.item()
    if self.hp_loops_number == self._loops_number:
      self.sigma *= cp.power(self.delta_sigma, 1/(self.hp_loops_number), dtype = cp.float32).item()
      self.delta_sigma = 1

  #set self.patence *= hp_param
  def update_sigma_heurestic(self,validation_score):
    if self.best_validation < validation_score:
      self.best_validation = validation_score
      self.should_heat_up = False
      self.delta_sigma *= self.sigma_drop
      self.iterations_without_improvment = 0
    else:
      self.iterations_without_improvment += 1
    if self.iterations_without_improvment >= self.patience:
      self.iterations_without_improvment = 0
      if self.should_heat_up:
        self.sigma = self.starting_sigma
        self.should_heat_up = False
      else:
        self.sigma *= self.sigma_drop ^ 100
        self.should_heat_up = True
    if self.hp_loops_number == self._loops_number:
      self.sigma *= cp.power(self.delta_sigma, 1/(self.hp_loops_number), dtype = cp.float32).item()
      self.delta_sigma = 1



     

  def fit(self, data, mu, lam, iterations): # mu is how many best samples from population, lam is how much we generate
    mean_act = cp.zeros(self.dimensionality)
    #constant
    mu //= self.hp_loops_number
    self.weights = cp.log(mu+1/2) - cp.log(cp.arange(1,mu+1))
    self.weights = self.weights/cp.sum(self.weights)
    mu_w = 1/cp.sum(self.weights**2)
    #dimension = 7840
    #c_sigma = (mu_w + 2)/(dimension + mu_w + 5)
    #d_sigma = 1 + 2*max([0,cp.sqrt((mu_w - 1)/(dimension + 1)) - 1]) + c_sigma #dampening parameter could probably be hyperparameter, wiki says it is close to 1 so whatever
    #c_covariance = (4 + mu_w/dimension)/(dimension + 4 + 2*mu_w/dimension) # c_covariance * 100 not working
    #c_1 = 2/(dimension**2)
    #c_mu = min([1-c_1,2*(mu_w - 2 + 1/mu_w)/(((dimension+2)**2)+mu_w)])

    c_1 = 2/(self.param_dimensionality**2)
    c_sigma = (mu_w + 2)/(self.param_dimensionality + mu_w + 5)
    d_sigma = 1 + 2*max([0,cp.sqrt((mu_w - 1)/(self.param_dimensionality + 1)) - 1]) + c_sigma #dampening parameter could probably be hyperparameter, wiki says it is close to 1 so whatever
    c_covariance = (4 + mu_w/self.param_dimensionality)/(self.param_dimensionality + 4 + 2*mu_w/self.param_dimensionality) # c_covariance * 100 not working
    c_mu = min([1-c_1,2*(mu_w - 2 + 1/mu_w)/(((self.param_dimensionality+2)**2)+mu_w)])

    
    file = open("PARAMS.txt", "w")
    file.write("c_1: " + str(c_1) + "\n")
    file.write("c_mu: " + str(c_mu) + "\n")
    file.write("c_sigma: " + str(c_sigma) + "\n")
    file.write("d_sigma: " + str(d_sigma) + "\n")
    file.write("c_covariance: " + str(c_covariance) + "\n")
    file.write("mu_w: " + str(mu_w) + "\n")
    file.close()
    
    alpha = 1.5
    #body 
    max_score = 0
    for i in range(iterations):
      train_scores, validation_scores = self.evaluate_func(self.population, data)
      sorted_indices = cp.argsort(-train_scores)
      mean_prev = mean_act.copy()
      self.population.parse_to_vector()
      if max_score < cp.max(train_scores):
        cp.save("drive/MyDrive/population/population.npy", population.matrix, allow_pickle=False)
      print("___bedzie udpate mean")
      mean_act = self.update_mean(train_scores,sorted_indices,mu) #we need to be vectorized here
      print("___bedzie logs log")
      self.logs.log([self.covariance_matrix,self.population.matrix,self.sigma,self.isotropic,self.anisotropic,
                      mean_prev,cp.max(train_scores), cp.max(validation_scores),mean_act-mean_prev])
      self.logs.plot()
      self.update_isotropic(mean_act,mean_prev,c_sigma,mu_w)
      c_s = self.compute_cs(alpha,c_1,c_covariance)
      self.update_anisotropic(mean_act,mean_prev,mu_w,c_covariance,alpha)
      self.update_covariance_matrix(c_1,c_mu,c_s,train_scores,sorted_indices,mu,mean_prev)
      if self.patience == None:
        self.update_sigma(c_sigma,d_sigma)
      else:
        self.update_sigma_heurestic(cp.max(validation_scores))
      self.population.sample(self.B_matrix, self.D_matrix, self.sigma, mean_act, lam)
      self.population.parse_from_vectors()
      if self._loops_number == self.hp_loops_number:
        self._loops_number = 0
      self._loops_number += 1
      file = open("LOGS.txt", "a")
      file.write(str(self._loops_number) + "\n\n")
      file.close()
    return self.population


  #MARIO
  def merge_populations(self,children_population,children_scores,parent_scores):
    #here population is in vectorized form!
    sorted_indices_children = cp.argsort(-children_scores)
    sorted_indices_parent = cp.argsort(-parent_scores)
    children_iterator,parent_iterator = 0,0
    scores = cp.zeros(self.population.population_size)
    dummy_population = cp.zeros(children_population.matrix.shape,dtype = cp.float32)
    #fuck that, no idea how to do it nicely, we have to create new dummy population each time
    for i in range(self.population.population_size):
      children_index = sorted_indices_children[children_iterator]
      parent_index = sorted_indices_parent[parent_iterator]
      children_score = children_scores[children_index]
      parent_score = parent_scores[parent_index]
      if children_score > parent_score:
        dummy_population[i] = cp.copy(children_population.matrix[children_index])
        scores[i] = children_scores[children_index]
        children_iterator += 1
      else:
        dummy_population[i] = cp.copy(self.population.matrix[parent_index])
        scores[i] = parent_scores[parent_index]
        parent_iterator += 1
    self.population.matrix = cp.copy(dummy_population) #motherfucker trick
    return scores,cp.zeros(self.population.population_size)


  def fit_mario_plus(self,data,mu,lam,iterations,children_population):
    mean_act = cp.zeros(self.dimensionality)
    mu //= self.hp_loops_number
    self.weights = cp.log(mu+1/2) - cp.log(cp.arange(1,mu+1))
    self.weights = self.weights/cp.sum(self.weights)
    mu_w = 1/cp.sum(self.weights**2)
    mu_w /= 10

    c_1 = 2/(self.param_dimensionality**2)
    c_sigma = (mu_w + 2)/(self.param_dimensionality + mu_w + 5)
    d_sigma = 1 + 2*max([0,cp.sqrt((mu_w - 1)/(self.param_dimensionality + 1)) - 1]) + c_sigma #dampening parameter could probably be hyperparameter, wiki says it is close to 1 so whatever
    c_covariance = (4 + mu_w/self.param_dimensionality)/(self.param_dimensionality + 4 + 2*mu_w/self.param_dimensionality) # c_covariance * 100 not working
    c_mu = min([1-c_1,2*(mu_w - 2 + 1/mu_w)/(((self.param_dimensionality+2)**2)+mu_w)])

    
    file = open("PARAMS.txt", "w")
    file.write("c_1: " + str(c_1) + "\n")
    file.write("c_mu: " + str(c_mu) + "\n")
    file.write("c_sigma: " + str(c_sigma) + "\n")
    file.write("d_sigma: " + str(d_sigma) + "\n")
    file.write("c_covariance: " + str(c_covariance) + "\n")
    file.write("mu_w: " + str(mu_w) + "\n")
    file.close()
    
    alpha = 1.5
    #preloop eval
    train_scores_parents, validation_scores_parents = self.evaluate_func(self.population, data)
    self.population.parse_to_vector()
    #body 
    for i in range(iterations):
      #sample children
      children_population.sample(self.B_matrix, self.D_matrix, self.sigma, mean_act, lam)
      children_population.parse_from_vectors()

      #eval children
      train_scores_children, validation_scores_children = self.evaluate_func(children_population, data)

      #merge children and parent
      children_population.parse_to_vector()
      train_scores_parents,validation_scores_parents = self.merge_populations(children_population,train_scores_children,train_scores_parents)
      train_scores = train_scores_parents
      validation_scores = validation_scores_parents
      #bad moment here assignment to self.population can crash (dunno)
      #standard part of loop
      sorted_indices = cp.argsort(-train_scores)
      mean_prev = mean_act.copy()
      

      #update
      print("___bedzie udpate mean")
      mean_act = self.update_mean(train_scores,sorted_indices,mu) #we need to be vectorized here
      print("___bedzie logs log")
      self.logs.log([self.covariance_matrix,self.population.matrix,self.sigma,self.isotropic,self.anisotropic,
                      mean_prev,cp.max(train_scores), cp.max(validation_scores),mean_act-mean_prev])
      self.logs.plot()
      self.update_isotropic(mean_act,mean_prev,c_sigma,mu_w)
      c_s = self.compute_cs(alpha,c_1,c_covariance)
      self.update_anisotropic(mean_act,mean_prev,mu_w,c_covariance,alpha)
      self.update_covariance_matrix(c_1,c_mu,c_s,train_scores,sorted_indices,mu,mean_prev)
      if self.patience == None:
        self.update_sigma(c_sigma,d_sigma)
      else:
        self.update_sigma_heurestic(cp.max(validation_scores))
      #finish
      file = open("LOGS.txt", "a")
      file.write(str(self._loops_number) + "\n\n")
      file.close()
    return self.population