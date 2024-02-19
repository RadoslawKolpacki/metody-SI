import numpy as np

def ackley(*args):
    n = len(args)
    sum1 = sum(x**2 for x in args)
    sum2 = sum(np.cos(2*np.pi*x) for x in args)
    return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.exp(1)

def generate_initial_solution(bounds):
    return [np.random.uniform(bounds[0], bounds[1]) for _ in range(len(bounds))]

def generate_neighbor(solution, bounds, step_size):
    return [np.clip(x + np.random.uniform(-step_size, step_size), bounds[0], bounds[1]) for x in solution]

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
bounds = [-5, 5]
max_iterations = 1000
initial_temperature = 100.0
final_temperature = 0.1
step_size = 0.5

best_solution = simulated_annealing(ackley, bounds, max_iterations, initial_temperature, final_temperature, step_size)
best_score = ackley(*best_solution)

print("Best solution:", best_solution)
print("Best score:", best_score)