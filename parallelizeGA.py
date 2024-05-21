import random
import numpy as np
import subprocess
from multiprocessing import Pool

def fitness_function(matrix):
    save_matrix_to_txt(matrix, 'matrix_weight.txt')
    try:
        output = subprocess.check_output(['java', '-jar', '/workspaces/SUB/SUB.jar'], text=True)
        out = output.strip().splitlines()[-1]
        out = eval(out.strip())
        out = 100 - out[0]
    except subprocess.CalledProcessError as e:
        print("Command execution failed:", e)
        return float('inf')  # Return a high fitness value if the subprocess fails
    return out

def save_matrix_to_txt(matrix, filename):
    np.savetxt(filename, matrix)

def generate_individual():
    return np.random.uniform(0, 1, size=(27, 3))

def generate_population(pop_size):
    return [generate_individual() for _ in range(pop_size)]

def calculate_fitness_individual(individual):
    return fitness_function(individual)

def calculate_fitness_population(population):
    with Pool() as pool:
        fitness_scores = pool.map(calculate_fitness_individual, population)
    return fitness_scores

def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [f / total_fitness for f in fitness_scores]
    return random.choices(population, weights=selection_probs, k=2)

def crossover(parents, crossover_rate):
    crossover_point = random.randint(1, len(parents[0]) - 1)
    if random.random() < crossover_rate:
        child1 = np.vstack((parents[0][:crossover_point], parents[1][crossover_point:]))
        child2 = np.vstack((parents[1][:crossover_point], parents[0][crossover_point:]))
    else:
        child1 = parents[0]
        child2 = parents[1]
    return child1, child2

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        individual += np.random.uniform(-0.1, 0.1, size=individual.shape)
    return individual

def genetic_algorithm(population_size, generations):
    population = generate_population(population_size)
    for generation in range(generations):
        fitness_scores = calculate_fitness_population(population)
        new_population = []
        for _ in range(population_size // 2):
            parents = select_parents(population, fitness_scores)
            offspring1, offspring2 = crossover(parents, crossover_rate=0.5)
            offspring1 = mutate(offspring1, mutation_rate=0.1)
            offspring2 = mutate(offspring2, mutation_rate=0.1)
            new_population.extend([offspring1, offspring2])
        
        combined_population = population + new_population
        sorted_population = sorted(combined_population, key=fitness_function)
        population = sorted_population[:population_size]
        
        print(f"Generation {generation + 1}, Best fitness: {fitness_function(population[0])}")
        
    return min(population, key=fitness_function)

best_solution = genetic_algorithm(population_size=10, generations=5)
print("Best solution:")
print(best_solution)
print("Fitness:", fitness_function(best_solution))