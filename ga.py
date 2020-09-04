import numpy
from math import pow

def cal_pop_fitness(p_naut, new_population,t,price_previous):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    #fitness = numpy.sum(pop*equation_inputs, axis=1)
    fitness=numpy.sum(new_population,axis=1)
    for i in range(new_population.shape[0]):
        alpha=(new_population[i,0]-new_population[i,1])/(new_population[i,3]-new_population[i,2])
        price=alpha+(p_naut-alpha)*pow((new_population[i,2]/abs(new_population[i,3])),t)
        if price_previous is None:
            fitness[i]=abs(price-p_naut)
        else:
            fitness[i]=abs(price-price_previous[i])
    return fitness

def price(p_naut, new_population,t):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    #fitness = numpy.sum(pop*equation_inputs, axis=1)
    price=numpy.sum(new_population,axis=1)
    for i in range(new_population.shape[0]):
        alpha=(new_population[i,0]-new_population[i,1])/(new_population[i,3]-new_population[i,2])
        price[i]=alpha+(p_naut-alpha)*pow((new_population[i,2]/abs(new_population[i,3])),t)
    return price

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    # parents = numpy.empty((num_parents, pop.shape[1]))
    # for parent_num in range(num_parents):
    #     max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
    #     max_fitness_idx = max_fitness_idx[0][0]
    #     parents[parent_num, :] = pop[max_fitness_idx, :]
    #     fitness[max_fitness_idx] = 99999999999
    
    parents = numpy.empty((num_parents,pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.min(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = 99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = numpy.random.uniform(-25, 25, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

def filter(offspring_mutation):
    i=0
    for _ in range(offspring_mutation.shape[0]):
        if offspring_mutation[i,3]>=0 or abs(offspring_mutation[i,3])<offspring_mutation[i,2]:
            offspring_mutation = numpy.delete(offspring_mutation, i,axis=0)
        else:
            i=i+1
    return offspring_mutation



