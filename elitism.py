import random
import numpy as np
import subprocess
from multiprocessing import Pool

def fitness_function(matrix):
    save_matrix_to_txt(matrix, 'matrix_weightNN.txt')
    try:
        output = subprocess.check_output(['java', '--add-opens', 'java.base/java.lang=ALL-UNNAMED', '-jar', 'SUB_desk.jar'],text=True)
        out = output.strip().splitlines()[-1]
        out = eval(out.strip())
        out = 96 - out[0]
    except subprocess.CalledProcessError as e:
        print("Command execution failed:", e)
        return float('inf') 
    return out

def save_matrix_to_txt(matrix, filename):
    np.savetxt(filename, matrix)

def generate_individual():
    n_in = 39
    n_hidden = 20
    n_out = 3
    
    limit_input_hidden = np.sqrt(6 / (n_in + n_hidden))
    limit_hidden_hidden = np.sqrt(6 / (n_hidden + n_hidden))
    limit_hidden_output = np.sqrt(6 / (n_hidden + n_out))
    
    matrix_input_hidden = np.random.uniform(-limit_input_hidden, limit_input_hidden, size=(n_hidden, n_in)).flatten()
    matrix_hidden_hidden = np.random.uniform(-limit_hidden_hidden, limit_hidden_hidden, size=(n_hidden, n_hidden)).flatten()
    matrix_hidden_output = np.random.uniform(-limit_hidden_output, limit_hidden_output, size=(n_out, n_hidden)).flatten()
    
    return np.concatenate([matrix_input_hidden,matrix_hidden_hidden, matrix_hidden_output])

def generate_population(pop_size):
    return [generate_individual() for _ in range(pop_size)]

def calculate_fitness_individual(individual):
    assert len(individual) == 39*20 + 20*20 + 20*3
    
    matrix_input_hidden = individual[:39*20].reshape((20, 39))
    matrix_hidden_hidden = individual[39*20:59*20].reshape((20, 20))
    matrix_hidden_output = individual[59*20:].reshape((3, 20))

    combined_matrix = np.concatenate((matrix_input_hidden.flatten(),matrix_hidden_hidden.flatten(), matrix_hidden_output.flatten()))
    
    return fitness_function(combined_matrix)

def calculate_fitness_population(population):
    with Pool(processes=1) as pool:
        fitness_scores = pool.map(calculate_fitness_individual, population)
    return fitness_scores

def select_parents(population, fitness_scores):
    inverted_fitness = [96 - f for f in fitness_scores]
    total_inverted_fitness = sum(inverted_fitness)
    selection_probs = [f / total_inverted_fitness for f in inverted_fitness]
    return random.choices(population, weights=selection_probs, k=2)

def crossover(parents, crossover_rate):
    crossover_point1 = random.randint(1, 780)
    crossover_point2 = random.randint(780, 1180)
    crossover_point3 = random.randint(1180, len(parents[0]) - 1)
    if random.random() < crossover_rate:
        child1 = np.concatenate((parents[0][:crossover_point1], parents[1][crossover_point1:780], parents[0][780:crossover_point2], parents[1][crossover_point2:1180], parents[0][1180:crossover_point3], parents[1][crossover_point3:]))
        child2 = np.concatenate((parents[1][:crossover_point1], parents[0][crossover_point1:780], parents[1][780:crossover_point2], parents[0][crossover_point2:1180], parents[1][1180:crossover_point3], parents[0][crossover_point3:]))
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
    return individual

def genetic_algorithm(population_size, generations):
    population = generate_population(population_size)
    best_fitness_per_generation = []
    fitness_scores = calculate_fitness_population(population)
    best_individual_index = np.argmin(fitness_scores)
    best_individual = population[best_individual_index]
    best_fitness = fitness_scores[best_individual_index]
    print(f"Generation {0}, Best fitness: {best_fitness}")
    for generation in range(generations):
        print(fitness_scores)
        new_population = []
        for _ in range(population_size // 2):
            parents = select_parents(population, fitness_scores)
            offspring1, offspring2 = crossover(parents, crossover_rate=0.5)
            offspring1 = polynomial_mutation(offspring1, mutation_rate=0.3)
            offspring2 = polynomial_mutation(offspring2, mutation_rate=0.3)
            new_population.extend([offspring1, offspring2])
        
        
        fitness_scores = calculate_fitness_population(population)
        if fitness_scores[0] > best_fitness:
            population = new_population
            population[0] = best_individual
            fitness_scores[0] = best_fitness
        best_individual_index = np.argmin(fitness_scores)
        best_individual = population[best_individual_index]
        
        
        best_fitness = fitness_scores[best_individual_index]
        best_fitness_per_generation.append(best_fitness)


        print(f"Generation {generation + 1}, Best fitness: {best_fitness}")
        if generation%10 == 9:
            combined_matrix = np.vstack(population)
            file_name = "final_populationNN.txt"
            np.savetxt(file_name, combined_matrix)
        if best_fitness == 0:
            break
    print(fitness_scores)
    np.savetxt("best_fitness_per_generationNN.txt", best_fitness_per_generation)
    np.savetxt("best_matrixNN.txt", best_individual, delimiter=',')
    return min(population, key=fitness_function)

if __name__ == '__main__':
    best_solution = genetic_algorithm(population_size=20, generations=300)
    print("Best solution:")
    print(best_solution)
    print("Fitness:", fitness_function(best_solution))