import numpy
import ga

# Inputs of the equation.
#equation_inputs = [4,-2,3.5,5,-11,-4.7]
p_naut=4.061

# Number of the weights we are looking to optimize.
#num_weights = 6
num_coeff=4

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 4
num_parents_mating =2

# Defining the population size.
pop_size = (sol_per_pop,num_coeff) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
#new_population = numpy.random.uniform(low=-2.0, high=2.0, size=pop_size)
matrix_A = numpy.random.uniform(low=972, high=2916, size=(sol_per_pop,1))
matrix_a = numpy.random.uniform(low=1622, high=4866, size=(sol_per_pop,1))
matrix_B = numpy.random.uniform(low=103, high=310, size=(sol_per_pop,1))
matrix_b = numpy.random.uniform(low=141, high=424, size=(sol_per_pop	,1))



for i in range(matrix_B.shape[0]):
	if matrix_B[i,0]>=matrix_b[i,0]:
		temp=matrix_b[i,0]
		matrix_b[i,0]=matrix_B[i,0]
		matrix_B[i,0]=temp

matrix_b=matrix_b*(-1)

new_population = numpy.concatenate((matrix_A,matrix_a,matrix_B,matrix_b),axis=1)

print(new_population)

num_generations=40
price_previous=None

for generation in range(num_generations):
	print("Generation : ", (generation+1))
	print("Initial generation : \n", (new_population))
	# Measing the fitness of each chromosome in the population.
	fitness= ga.cal_pop_fitness(p_naut,new_population,(generation+1),price_previous)
	# Selecting the best parents in the population for mating.
	parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)
	print("Selected parents to mate : \n", (parents))
	# Generating next generation using crossover.
	offspring_crossover = ga.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_coeff))
	print("After crossover offspring: \n", (offspring_crossover))
	# Adding some variations to the offsrping using mutation.
	offspring_mutation = ga.mutation(offspring_crossover)
	print("After mutation : \n", (offspring_mutation))
	#rejecting disformed offspring
	#goodShape_offspring=ga.filter(offspring_mutation)
	# Creating the new population based on the parents and offspring.
	new_population[0:parents.shape[0], :] = parents
	new_population[parents.shape[0]:, :] = offspring_mutation
	print("New generation : \n", (new_population))
	# new_population=numpy.concatenate((parents,goodShape_offspring))
	# population=numpy.concatenate((parents,goodShape_offspring))
	# print("mutants : ", (population))
	# The best result in the current iteration.
	price_previous=ga.price(p_naut,new_population,generation+1)
	fitness=ga.cal_pop_fitness(p_naut,new_population,generation+2,price_previous)
	max_fitness_idx = numpy.where(fitness == numpy.min(fitness))
	max_fitness_idx = max_fitness_idx[0][0]
	print("Best result : ", new_population[max_fitness_idx, :])
		


# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(p_naut, new_population,5,price_previous)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])
