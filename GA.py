import random
import numpy as np
import subprocess

def fitness_function(matrix):
    save_matrix_to_txt(matrix, 'matrix_weight.txt')
    try:
        output = subprocess.check_output(['java', '-jar', '/workspaces/SUB/SUB.jar'], text=True)
        out = output.strip().splitlines()[-1]
        out = eval(out.strip())
        out = 100-out[0]
        #print(out)
    except subprocess.CalledProcessError as e:
        print("Command execution failed:", e)
    return out

def save_matrix_to_txt(matrix, filename):
    with open(filename, 'w') as file:
        for row in matrix:
            file.write(' '.join(map(str, row)) + '\n')

# 初始化個體，這裡假設每個個體是一個 39x3 的矩陣，元素取值在 -10 到 10 之間
def generate_individual():
    return np.random.uniform(-10, 10, size=(39, 3))

# 生成初始群體
def generate_population(pop_size):
    return [generate_individual() for _ in range(pop_size)]

# 計算每個個體的適應度
def calculate_fitness(population):
    return [fitness_function(individual) for individual in population]

# 選擇適應度較高的個體作為父母個體
def select_parents(population, fitness_scores):
    return random.choices(population, weights=fitness_scores, k=2)

# 交叉（交配）操作，這裡我們使用簡單的交叉點交配
def crossover(parents):
    crossover_point = random.randint(1, len(parents[0])-1)
    child1 = np.vstack((parents[0][:crossover_point], parents[1][crossover_point:]))
    child2 = np.vstack((parents[1][:crossover_point], parents[0][crossover_point:]))
    return child1, child2

# 突變操作，這裡我們使用簡單的微小突變
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        individual += np.random.uniform(-0.1, 0.1, size=individual.shape)  # 對每個元素添加一個在[-0.1, 0.1]範圍內的隨機值
    return individual

# 基因演算法主函數
def genetic_algorithm(population_size, generations):
    population = generate_population(population_size)
    for _ in range(generations):
        fitness_scores = calculate_fitness(population)
        new_population = []
        for _ in range(population_size // 2):
            parents = select_parents(population, fitness_scores)
            offspring1, offspring2 = crossover(parents)
            offspring1 = mutate(offspring1, mutation_rate=0.1)
            offspring2 = mutate(offspring2, mutation_rate=0.1)
            new_population.extend([offspring1, offspring2])
        
        # 將新生成的族群和原先的族群合併
        combined_population = population + new_population
        
        # 根據適應度分數對合併後的族群進行排序，選擇適應度較好的一半
        sorted_population = sorted(combined_population, key=fitness_function)
        population = sorted_population[:population_size]
        
    # 返回最終的最佳解答
    return min(population, key=fitness_function)

# 測試基因演算法
best_solution = genetic_algorithm(population_size=100, generations=100)
print("Best solution:")
print(best_solution)
print("Fitness:", fitness_function(best_solution))
