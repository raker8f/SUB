import random
import numpy as np
import subprocess
from multiprocessing import Pool

def fitness_function(matrix):
    save_matrix_to_txt(matrix, 'order.txt')
    fit = 0
    for i in range(5):
        try:
            output = subprocess.check_output(['java', '--add-opens', 'java.base/java.lang=ALL-UNNAMED', '-jar', 'SUB_desk.jar'], text=True)
            out = output.strip().splitlines()[-1]
            out = eval(out.strip())
            out = 96 - out[0]
            fit = fit+out
        except subprocess.CalledProcessError as e:
            print("Command execution failed:", e)
            fit = 1000
            return float('inf')  # Return a high fitness value if the subprocess fails
    return fit

def save_matrix_to_txt(matrix, filename):
    np.savetxt(filename, matrix)

def generate_individual():
    scan_order = list(range(2, 14))  # Assuming initial scan_order
    random.shuffle(scan_order)
    back_to_save = [random.choice([0, 1]) for _ in range(12)]
    return scan_order + back_to_save

def generate_population(pop_size):
    return [generate_individual() for _ in range(pop_size)]

def calculate_fitness_individual(individual):
    return fitness_function(individual)

def calculate_fitness_population(population):
    with Pool(processes=1) as pool:
        fitness_scores = pool.map(calculate_fitness_individual, population)
    return fitness_scores

def select_parents(population, fitness_scores):
    inverted_fitness = [480 - f for f in fitness_scores]
    total_inverted_fitness = sum(inverted_fitness)
    selection_probs = [f / total_inverted_fitness for f in inverted_fitness]
    return random.choices(population, weights=selection_probs, k=2)

def pox_crossover(order1, order2):
    size = len(order1)
    cxpoint1 = random.randint(0, size - 1)
    cxpoint2 = random.randint(0, size - 1)
    start, end = min(cxpoint1, cxpoint2), max(cxpoint1, cxpoint2)

    child1, child2 = [-1] * size, [-1] * size

    # Copy the crossover segment
    for i in range(start, end + 1):
        child1[i] = order1[i]
        child2[i] = order2[i]

    def fill_rest(child, parent):
        child_idx, parent_idx = end + 1, end + 1
        while -1 in child:
            if parent[parent_idx % size] not in child:
                child[child_idx % size] = parent[parent_idx % size]
                child_idx += 1
            parent_idx += 1

    fill_rest(child1, order2)
    fill_rest(child2, order1)

    return child1, child2

def crossover(parents, crossover_rate):
    order1 = parents[0][:12]
    order2 = parents[1][:12]
    save1 = parents[0][12:]
    save2 = parents[1][12:]
    crossover_point = random.randint(1, len(save1) - 1)
    if random.random() < crossover_rate:
        child_save1 = np.concatenate((save1[:crossover_point], save2[crossover_point:]))
        child_save2 = np.concatenate((save2[:crossover_point], save1[crossover_point:]))
        child_order1, child_order2 = pox_crossover(order1, order2)
    else:
        child_save1 = save1
        child_save2 = save2
        child_order1 = order1
        child_order2 = order2
    child1 = np.concatenate((child_order1, child_save1))
    child2 = np.concatenate((child_order2, child_save2))
    return child1, child2

def two_swap_mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
        size = len(individual)
        pos1 = random.randint(0, size - 1)
        pos2 = random.randint(0, size - 1)
        individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
    return individual

def flip_mutation(individual, mutation_prob):
    mutated_individual = individual[:]
    for i in range(len(mutated_individual)):
        if random.random() < mutation_prob:
            mutated_individual[i] = 1 - mutated_individual[i] 
    return mutated_individual

def genetic_algorithm(population_size, generations):
    population = generate_population(population_size)
    best_fitness_per_generation = []
    fitness_scores = calculate_fitness_population(population)
    for generation in range(generations):
        print(fitness_scores)
        new_population = []
        for _ in range(population_size // 2):
            parents = select_parents(population, fitness_scores)
            child1, child2 = crossover(parents, crossover_rate=0.5)
            
            offspring1 = two_swap_mutation(child1[:12], mutation_rate=0.2)
            offspring1 = np.concatenate((offspring1, flip_mutation(child1[12:], mutation_prob=0.1)))
            
            offspring2 = two_swap_mutation(child2[:12], mutation_rate=0.2)
            offspring2 = np.concatenate((offspring2, flip_mutation(child2[12:], mutation_prob=0.1)))

            new_population.extend([offspring1, offspring2])
        
        new_fitness_scores = calculate_fitness_population(new_population)
        all_population = population + new_population
        fitness_scores += new_fitness_scores
        
        sorted_indices = np.argsort(fitness_scores)
        fitness_scores = [fitness_scores[i] for i in sorted_indices[:population_size]]
        population = [all_population[i] for i in sorted_indices[:population_size]]
        
        best_fitness = fitness_scores[0]
        best_fitness_per_generation.append(best_fitness)
        print(f"Generation {generation + 1}, Best fitness: {best_fitness}")
        if generation % 10 == 9:
            combined_matrix = np.vstack(population)
            file_name = "final_population_level2.txt"
            np.savetxt(file_name, combined_matrix)
        if best_fitness == 0:
            break

    np.savetxt("best_fitness_per_generation_level2.txt", best_fitness_per_generation, delimiter=',')
    np.savetxt("best_matrix_level2.txt", population[0], delimiter=',')
    return population[0], fitness_scores[0]

if __name__ == '__main__':
    best_solution, Fitness = genetic_algorithm(population_size=10, generations=50)
    print("Best solution:")
    print(best_solution)
    print("Fitness:", Fitness)
