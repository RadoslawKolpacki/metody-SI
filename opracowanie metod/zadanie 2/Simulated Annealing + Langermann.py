import numpy as np

def langermann(x, y):
    A = [[3, 5], [5, 2], [2, 1], [1, 4]]
    C = [1, 2, 5, 2]
    M = len(C)
    result = 0

    for i in range(M):
        term1 = (x - A[i][0])**2
        term2 = (y - A[i][1])**2
        term3 = np.cos(np.pi * term1 + term2)
        result += C[i] * term3

    return -result

def generate_initial_solution(bounds):
    x = np.random.uniform(bounds[0], bounds[1])
    y = np.random.uniform(bounds[0], bounds[1])
    return [x, y]

def generate_neighbor(solution, bounds, step_size):
    x, y = solution
    min_x = max(bounds[0], x - step_size)
    max_x = min(bounds[1], x + step_size)
    min_y = max(bounds[0], y - step_size)
    max_y = min(bounds[1], y + step_size)
    new_x = np.random.uniform(min_x, max_x)
    new_y = np.random.uniform(min_y, max_y)
    return [new_x, new_y]

def acceptance_probability(current_score, new_score, temperature):
    if new_score < current_score:
        return 1.0
    return np.exp((current_score - new_score) / temperature)

def simulated_annealing(objective_func, bounds, max_iterations, initial_temperature, final_temperature, step_size):
    current_solution = generate_initial_solution(bounds)
    best_solution = current_solution
    current_score = objective_func(*current_solution)
    best_score = current_score
    temperature = initial_temperature
    iteration = 0

    while temperature > final_temperature and iteration < max_iterations:
        new_solution = generate_neighbor(current_solution, bounds, step_size)
        new_score = objective_func(*new_solution)
        ap = acceptance_probability(current_score, new_score, temperature)

        if ap > np.random.random():
            current_solution = new_solution
            current_score = new_score

        if new_score < best_score:
            best_solution = new_solution
            best_score = new_score

        temperature *= 0.9  # Cooling schedule
        iteration += 1

    return best_solution

# Example usage:
bounds = [0, 10]
max_iterations = 1000
initial_temperature = 100.0
final_temperature = 0.1
step_size = 0.5

best_solution = simulated_annealing(langermann, bounds, max_iterations, initial_temperature, final_temperature, step_size)
best_score = langermann(*best_solution)

print("Best solution:", best_solution)
print("Best score:", best_score)

