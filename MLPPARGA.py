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
    n_in = 63
    n_hidden = 10
    n_out = 3
    
    limit_input_hidden = np.sqrt(6 / (n_in + n_hidden))
    limit_hidden_output = np.sqrt(6 / (n_hidden + n_out))
    
    matrix_input_hidden = np.random.uniform(-limit_input_hidden, limit_input_hidden, size=(n_hidden, n_in)).flatten()
    matrix_hidden_output = np.random.uniform(-limit_hidden_output, limit_hidden_output, size=(n_out, n_hidden)).flatten()
    
    return np.concatenate([matrix_input_hidden, matrix_hidden_output])

def generate_population(pop_size):
    return [generate_individual() for _ in range(pop_size)]

def calculate_fitness_individual(individual):
    assert len(individual) == 63*10 + 10*3, "個體長度不正確"
    
    matrix_input_hidden = individual[:63*10].reshape((10, 63))
    matrix_hidden_output = individual[63*10:].reshape((3, 10))

    combined_matrix = np.concatenate((matrix_input_hidden.flatten(), matrix_hidden_output.flatten()))
    
    return fitness_function(combined_matrix)

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
        child1 = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
        child2 = np.concatenate((parents[1][:crossover_point], parents[0][crossover_point:]))
    else:
        child1 = parents[0]
        child2 = parents[1]
    return child1, child2

def polynomial_mutation(individual, mutation_rate, eta=20):
    size = individual.shape
    for i in range(size[0]):
        if random.random() < mutation_rate:
            u = random.random()
            delta = (2*u)**(1/(eta+1)) - 1 if u < 0.5 else 1 - (2*(1-u))**(1/(eta+1))
            individual[i] += delta
            individual[i] = min(max(individual[i], 0), 1)
    return individual

def genetic_algorithm(population_size, generations):
    population = generate_population(population_size)
    best_fitness_per_generation = []
    for generation in range(generations):
        fitness_scores = calculate_fitness_population(population)
        new_population = []
        for _ in range(population_size // 2):
            parents = select_parents(population, fitness_scores)
            offspring1, offspring2 = crossover(parents, crossover_rate=0.5)
            offspring1 = polynomial_mutation(offspring1, mutation_rate=0.1)
            offspring2 = polynomial_mutation(offspring2, mutation_rate=0.1)
            new_population.extend([offspring1, offspring2])
        
        combined_population = population + new_population
        sorted_population = sorted(combined_population, key=fitness_function)
        population = sorted_population[:population_size]
        
        best_fitness = fitness_function(population[0])
        best_fitness_per_generation.append(best_fitness)

        print(f"Generation {generation + 1}, Best fitness: {best_fitness}")
        if generation%100 == 99:
            combined_matrix = np.vstack(population)
            file_name = "final_population.txt"
            np.savetxt(file_name, combined_matrix)

    np.savetxt("best_fitness_per_generation.txt", best_fitness_per_generation)
    return min(population, key=fitness_function)

best_solution = genetic_algorithm(population_size=20, generations=5)
print("Best solution:")
print(best_solution)
print("Fitness:", fitness_function(best_solution))
