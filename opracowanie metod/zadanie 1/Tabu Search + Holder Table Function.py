import numpy as np

def holder_table(x, y):
    part1 = np.sin(x) * np.cos(y)
    part2 = np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / np.pi))
    return -np.abs(part1 * part2)

def generate_initial_solution(bounds):
    x = np.random.uniform(bounds[0], bounds[1])
    y = np.random.uniform(bounds[0], bounds[1])
    return [x, y]

def generate_neighborhood(solution, bounds, step_size):
    x, y = solution
    min_x = max(bounds[0], x - step_size)
    max_x = min(bounds[1], x + step_size)
    min_y = max(bounds[0], y - step_size)
    max_y = min(bounds[1], y + step_size)
    new_x = np.random.uniform(min_x, max_x)
    new_y = np.random.uniform(min_y, max_y)
    return [[new_x, new_y]]  # Return as a list of coordinates

def tabu_search(objective_func, bounds, max_iterations, tabu_size, step_size):
    current_solution = generate_initial_solution(bounds)
    best_solution = current_solution
    tabu_list = [current_solution]
    iteration = 0

    while iteration < max_iterations:
        neighborhood = generate_neighborhood(current_solution, bounds, step_size)
        best_neighbor = None
        best_neighbor_score = float('inf')

        for neighbor in neighborhood:
            if neighbor not in tabu_list:
                neighbor_score = objective_func(*neighbor)
                if neighbor_score < best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = neighbor_score

        if best_neighbor is not None:
            current_solution = best_neighbor
            tabu_list.append(current_solution)

            if len(tabu_list) > tabu_size:
                tabu_list = tabu_list[1:]

            if objective_func(*current_solution) < objective_func(*best_solution):
                best_solution = current_solution

        iteration += 1

    return best_solution

# Example usage:
bounds = [-10, 10]
max_iterations = 100
tabu_size = 10
step_size = 1

best_solution = tabu_search(holder_table, bounds, max_iterations, tabu_size, step_size)
best_score = holder_table(*best_solution)

print("Best solution:", best_solution)
print("Best score:", best_score)
