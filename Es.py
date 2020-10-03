import numpy as np
import copy
import cupy as cp
from Engeneeringthesis.sigmas import Sigmas_Neural_Network
from Engeneeringthesis.NeuralNetwork import Neural_Network

no_debug = 1
basic_debug_mode = 2
super_debug_mode = 3
only_interesting = 5
DEBUG_MODE = only_interesting

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
def cuda_memory_clear():
    print("_total_bytes_before", mempool.total_bytes())
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()          
    print("_total_bytes_after", mempool.total_bytes())  

def generate_population():
    #300 - liczebnosc populacji
    raise exception("NEI WOLNO")
    mario_net = Neural_Network([('linear',(300, 28*28, 10))])
    mario_net_sigmas = sigmas_Mario_net([('linear',(300, 28*28, 10))])
    return (mario_net, mario_net_sigmas)

def best_population(population, population_scores, sigmas, population_size = 200000, children_size = 200000):
    if DEBUG_MODE % basic_debug_mode == 0:
      print("__best_population_start")
    population_size = population.layers[0][1].shape[0]
    children_size = population_size
    #population.move_to_cpu()
    new_marionet = Neural_Network(population.population_size,  population.input_size, population.input_layers)
    new_sigmas = Sigmas_Neural_Network(sigmas.sigmas_size,  sigmas.input_size, sigmas.input_layers)
    
    #new_marionet.move_to_cpu()
    for i in range(children_size):
        #participants = cp.random.choice( a = population_size, size = 2, replace = False)
        #chosen_one = cp.argmax(population_scores[participants])
        chosen_one = cp.argmax(population_scores)
        new_marionet.replace_individual(i, population.get_individual(chosen_one))
        new_sigmas.replace_individual(i, sigmas.get_individual(chosen_one))
    if DEBUG_MODE % basic_debug_mode == 0:
      print("__best_population_stop")
    return new_marionet, new_sigmas

def mutate_sigmas(sigmas, mutation_parameter_individual, mutation_parameter_coordinate):
  if DEBUG_MODE % basic_debug_mode == 0:
    print("SIGMAS MUTATION START", sigmas.layers_sigmas[0][1].shape)
  flag = 0
  random_individual_mutation = cp.random.normal(loc = 0., scale = mutation_parameter_individual, size = (sigmas.layers_sigmas[0][1].shape[0], 1, 1, 1, 1))
  for layer in sigmas.layers_sigmas:
      if layer[0]=='linear' and flag==0:
          flag=1
          random_individual_mutation = random_individual_mutation.reshape(sigmas.layers_sigmas[0][1].shape[0], 1 ,1)
      random_weight_mutation = cp.random.normal(loc = 0., scale = mutation_parameter_coordinate, size = layer[1].shape)
      layer[1] *= cp.exp(random_weight_mutation + random_individual_mutation).astype(cp.float32)
  if DEBUG_MODE % basic_debug_mode == 0:
    print("SIGMAS MUTATION STOP", sigmas.layers_sigmas[0][1].shape)

def mutate_network(population, sigmas):
  for j in range(len(population.layers)):
    population.layers[j][1] += cp.random.normal(0, sigmas.layers_sigmas[j][1], sigmas.layers_sigmas[j][1].shape)


def mutate_population(parents, sigmas, mutation_parameter_individual, mutation_parameter_coordinate):
    if DEBUG_MODE % basic_debug_mode == 0:
      print("__mutate_population_start")
    mutate_sigmas(sigmas, mutation_parameter_individual, mutation_parameter_coordinate)
    mutate_network(parents, sigmas)
    return parents, sigmas

def gen_new_population(population, population_sigmas, children, children_sigmas, population_scores, children_scores):
    if DEBUG_MODE % basic_debug_mode == 0:
      print("__gen_new_population_start")
    population_size = population.layers[0][1].shape[0]
    population_argsorted_scores = cp.argsort(-population_scores)
    children_argsorted_scores = cp.argsort(-children_scores)
    new_population = Neural_Network(population.population_size,  population.input_size, population.input_layers)
    new_sigmas = Sigmas_Neural_Network(population_sigmas.sigmas_size,  population_sigmas.input_size, population_sigmas.input_layers)
    

    new_scores = cp.zeros(population_size, dtype = cp.float32)

    population_pointer = 0 
    children_pointer = 0

    for i in range(population_size):
        if population_scores[population_argsorted_scores[population_pointer]] > children_scores[children_argsorted_scores[children_pointer]]:
            new_population.replace_individual(i, population.get_individual(population_argsorted_scores[population_pointer]))
            new_sigmas.replace_individual(i, population_sigmas.get_individual(population_argsorted_scores[population_pointer]))
            new_scores[i] = population_scores[population_argsorted_scores[population_pointer]]
            population_pointer+=1
        else:
            new_population.replace_individual(i, children.get_individual(children_argsorted_scores[children_pointer]))
            new_sigmas.replace_individual(i, children_sigmas.get_individual(children_argsorted_scores[children_pointer]))
            new_scores[i] = children_scores[children_argsorted_scores[children_pointer]]
            children_pointer+=1    

  
    #new_population.move_to_gpu()
    if DEBUG_MODE % basic_debug_mode == 0:
      print("__gen_new_population_stop")
    return new_population, new_sigmas, new_scores

def ES(population, sigmas, train_ds, iter_num=2, mutation_parameter_individual=.0001, mutation_parameter_coordinate=.0001):
    global best_indivudal_cupy
    population_size = population.layers[0][1].shape[0]
    children_size = population_size
    population_scores = evaluate_population(population, train_ds)
    best_results = []
    mean_results = []
    min_results = []

    sigmas_maxes = []
    sigmas_mins = []
    sigmas_means = []

    children_maxes = []
    children_mins = []
    children_means = []
    children_diff_from_best = []


    best_results.append(cp.max(population_scores))
    mean_results.append(cp.mean(population_scores))
    min_results.append(cp.min(population_scores))
    
    for i in range(iter_num):
        parents, parents_sigmas = best_population(population, population_scores, sigmas)
        children, children_sigmas = mutate_population(parents, sigmas, mutation_parameter_individual, mutation_parameter_coordinate)
        #children.move_to_gpu()
        cuda_memory_clear()
        children_scores = evaluate_population(children, train_ds)
        clear_output()
        print("BEST CHILDREN RESULT ", cp.max(children_scores))
        best_results.append(cp.max(children_scores))
        mean_results.append(cp.mean(children_scores))
        min_results.append(cp.min(children_scores))

        children_maxes.append(cp.max(population.layers[0][1][0]))
        children_mins.append(cp.min(population.layers[0][1][0]))
        children_means.append(cp.mean(population.layers[0][1][0]))
        children_diff_from_best.append(cp.mean(cp.abs(population.layers[0][1][0] - best_indivudal_cupy)))

        sigmas_maxes.append(cp.max(sigmas.layers_sigmas[0][1][0]))
        sigmas_mins.append(cp.min(sigmas.layers_sigmas[0][1][0]))
        sigmas_means.append(cp.mean(sigmas.layers_sigmas[0][1][0]))


        fig, axes = plt.subplots(2,2, figsize = (14,10))
        axes[0][0].plot(np.array(best_results), color = 'r')
        axes[0][0].plot(np.array(mean_results), color = 'g')
        axes[0][0].plot(np.array(min_results),  color  = 'b')
        axes[0][0].grid(True)
        axes[0][0].set_title('Results:')


        axes[0][1].plot(np.array(children_maxes), color = 'r')
        axes[0][1].plot(np.array(children_mins), color = 'g')
        axes[0][1].plot(np.array(children_means),  color  = 'b')
        axes[0][0].grid(True)
        axes[0][1].set_title('Weights:')

        #axes[1][0].plot(np.array(sigmas_maxes), color = 'r')
        axes[1][0].plot(np.array(sigmas_mins), color = 'g')
        axes[1][0].plot(np.array(sigmas_means),  color  = 'b')
        axes[0][0].grid(True)
        axes[1][0].set_title("Sigmas weights:")

        axes[1][1].plot(np.array(children_diff_from_best), color = 'r')
        axes[0][0].grid(True)
        axes[1][1].set_title('Distance from classically trained network')
        fig.tight_layout()
        plt.show()
        

        #population.move_to_gpu()
        #children.move_to_gpu()
        cuda_memory_clear()
        population, sigmas, population_scores = gen_new_population(population, sigmas, children, children_sigmas, population_scores, children_scores)
        cuda_memory_clear()
    return population, sigmas
    #return kozak_scores, mean_scores, worst_scores


def brute_dot(temp, lin):
    ret_temp = cp.zeros((temp.shape[0], lin.shape[2]))
    for i in range(temp.shape[0]):
        ret_temp[i] = cp.dot(temp[i], lin[i])
    return ret_temp


def evaluate_population(population, train_ds):
    create_input_time = 0
    preds_time = 0
    points_count_time = 0
    j  = 0
    if DEBUG_MODE % basic_debug_mode == 0:
      print("___EVALUATE_POPULATION_START")
    #scores = np.zeros(population.layers[0][1].shape[0], dtype = np.uint32)
    scores = cp.zeros(population.population_size, dtype = cp.uint32)
    for image, label in zip(cp.array(train_ds['image']), cp.array(train_ds['label'])):
        start = time.time()
        image = image.flatten()
        create_input_time += time.time() - start
        start = time.time()
        preds = population.forward(image)
        preds_time += time.time() - start
        start = time.time()
        #scores += cp.asnumpy(preds == label)
        scores += preds == label
        points_count_time += time.time() - start
        j += 1
      
    if DEBUG_MODE % basic_debug_mode == 0:
      print("___EVALUATE_POPULATION_STOP", "create_input_time: ", create_input_time, "preds_time:", preds_time,
          "points_count_time: ", points_count_time, "\n best result: ", np.max(cp.asnumpy( scores)),
          "mean socre: ", np.mean(cp.asnumpy( scores)), "min score: ", np.min(cp.asnumpy( scores))) 
    if DEBUG_MODE % only_interesting == 0:
      print("best result: ", np.max(cp.asnumpy( scores)), "mean socre: ", np.mean(cp.asnumpy( scores)), "min score: ", np.min(cp.asnumpy( scores)))

    return scores