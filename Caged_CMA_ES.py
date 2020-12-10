class Caged_CMA_ES():
  def __init__(self,population,sigma,evaluate_func, logs, number_of_cages, dimensionalities):
    self.cages = []
    self.logs = logs
    self.number_of_cages = number_of_cages
    self.population = population
    self.dimensionalities = dimensionalities
    self.dimensionality = sum(self.dimensionalities)
    self.evaluate_func = evaluate_func

    for i in range(number_of_cages):
      self.cages.append(CMA_ES(population, sigma, evaluate_func, logs, dimensionalities[i], i))

     
  def set_mean_act(self):
    ret_mean_act = []
    for i in range(number_of_cages):
      ret_mean_act.append(cp.zeros(self.dimensionalities[i], dtype = cp.float32))
    return ret_mean_act

  def update_mean(self, scores,sorted_indices,mu):
    ret_list = []
    for i in range(number_of_cages):
      ret_list.append(self.cages[i].update_mean(scores,sorted_indices,mu))
    return ret_list
    

  def set_cs(self):
    ret_list = []
    for i in range(self.number_of_cages):
      ret_list.append(0)
    return ret_list

  def find_division(self, number):
    for i in range(int(np.sqrt(number)), 2, -1):
      if number % i == 0:
        return (i , number // i)
      

  def parse_log_args(self, mean_act, scores):
    horizontal_stacks = []
    horizontal, vertical = self.find_division(self.number_of_cages)
    for i in range(vertical):
      horizontal_stacks.append(np.array(cp.asnumpy(self.cages[i*horizontal].covariance_matrix)))
      horizontal_stacks[i] = np.hstack((horizontal_stacks[i], np.zeros((self.cages[i*horizontal].covariance_matrix.shape[0], 1)) - 1))
      for j in range(1, horizontal):
        horizontal_stacks[i] = np.hstack((horizontal_stacks[i], cp.asnumpy(self.cages[i*horizontal + j].covariance_matrix)))
        horizontal_stacks[i] = np.hstack((horizontal_stacks[i], np.zeros((self.cages[i*horizontal + j].covariance_matrix.shape[0], 1)) - 1))
    
    cov = horizontal_stacks[0]
    cov = np.vstack((cov, np.zeros((1, horizontal_stacks[0].shape[1])) - 1 ))
    for i in range(1, vertical):
      cov = np.vstack((cov, horizontal_stacks[i]))
      cov = np.vstack((cov, np.zeros((1, horizontal_stacks[0].shape[1])) - 1 ))
    
    sigmas = []
    for i in range(self.number_of_cages):
      sigmas.append(self.cages[i].sigma)
    sigmas = np.array(sigmas)

    isotropic = np.array([])
    anisotropic = np.array([])
    for i in range(self.number_of_cages):
      isotropic = np.concatenate((isotropic, cp.asnumpy(self.cages[i].isotropic)))
      anisotropic = np.concatenate((anisotropic, cp.asnumpy(self.cages[i].anisotropic)))

    mean = np.array([])
    for i in range(self.number_of_cages):
      mean = np.concatenate((mean, cp.asnumpy(mean_act[i])))
    return [cov, sigmas, isotropic, anisotropic, mean, cp.max(scores)]


  def fit(self, data, mu, lam, iterations): # mu is how many best samples from population, lam is how much we generate
    for i in range(self.number_of_cages):
      self.cages[i].weights = cp.log(mu+1/2) - cp.log(cp.arange(1,mu+1))
      self.cages[i].weights = self.cages[i].weights/cp.sum(self.cages[i].weights)

    
    mu_w = 1/cp.sum(self.cages[0].weights**2)

    weights = cp.log(mu+1/2) - cp.log(cp.arange(1,mu+1))
    weights = weights/cp.sum(weights)
    problem_mu_w = 1/cp.sum(weights**2)
     
    c_1, c_sigma, d_sigma, c_covariance, c_mu = np.zeros(self.number_of_cages), np.zeros(self.number_of_cages), np.zeros(self.number_of_cages), np.zeros(self.number_of_cages), np.zeros(self.number_of_cages)

    alpha = 1.5
    for i in range(self.number_of_cages):
      c_1[i] = 2/(self.dimensionalities[i]**2)
      c_sigma[i] = (mu_w + 2)/(self.dimensionality + mu_w + 5)
      d_sigma[i] = 1 + 2*max([0,cp.sqrt((mu_w - 1)/(self.dimensionality + 1)) - 1]) + c_sigma[i] #dampening parameter could probably be hyperparameter, wiki says it is close to 1 so whatever
      c_covariance[i] = (4 + mu_w/self.dimensionalities[i])/(self.dimensionalities[i] + 4 + 2*mu_w/self.dimensionalities[i]) # c_covariance * 100 not working
      c_mu[i] = min([1-c_1[i],2*(mu_w - 2 + 1/mu_w)/(((self.dimensionalities[i]+2)**2)+mu_w)])
      
      
    
    
    mean_act = self.set_mean_act()
    mean_prev = self.set_mean_act()
    c_s = self.set_cs()

    file = open("PARAMS.txt", "w")
    file.write("c_1: " + str(c_1) + "\n \n")
    file.write("c_mu: " + str(c_mu) + "\n")
    file.write("c_sigma: " + str(c_sigma) + "\n \n")
    file.write("d_sigma: " + str(d_sigma) + "\n \n")
    file.write("c_covariance: " + str(c_covariance) + "\n")
    file.write("mu_w: " + str(mu_w) + "\n")
    file.write("problem_mu_w: " + str(problem_mu_w) + "\n")
    
    file.close()

    #body 
    for i in range(iterations):
      scores = self.evaluate_func(self.population, data)
      print(cp.max(scores))
      sorted_indices = cp.argsort(-scores)
      for j in range(len(mean_prev)):
        mean_prev[j] = mean_act[j].copy() #maybe deepcopy
      self.population.parse_to_vector()
      mean_act = self.update_mean(scores,sorted_indices,mu) #we need to be vectorized here
      self.logs.log(self.parse_log_args(mean_act, scores))
      self.logs.plot()
      for j in range(self.number_of_cages):
        self.cages[j].update_isotropic(mean_act[j],mean_prev[j],c_sigma[j],problem_mu_w)
        c_s[j] = self.cages[j].compute_cs(alpha,c_1[j],c_covariance[j])
        self.cages[j].update_anisotropic(mean_act[j],mean_prev[j],mu_w,c_covariance[j],alpha)
        self.cages[j].update_covariance_matrix(c_1[j],c_mu[j],c_s[j],scores,sorted_indices,mu,mean_prev[j])
        self.cages[j].update_sigma(c_sigma[j],d_sigma[j])

      
      Bs = []
      Ds = []
      sigmas = []
      for j in range(self.number_of_cages):
        Bs.append(self.cages[j].B_matrix)
        Ds.append(self.cages[j].D_matrix)
        sigmas.append(self.cages[j].sigma)
      self.population.caged_sample(Bs, Ds, sigmas, mean_act, lam)
      covariances = []
      sigams = []
      
      self.population.parse_from_vectors()
    return self.population