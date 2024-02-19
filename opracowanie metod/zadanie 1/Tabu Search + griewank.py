import numpy as np

def griewank(*args):
    part1 = sum(x**2 / 4000 for x in args)
    part2 = sum(np.cos(x / np.sqrt(i+1)) for i, x in enumerate(args))
    return 1 + part1 - part2

def generate_initial_solution(bounds, dimensions):
    return [np.random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]

def generate_neighborhood(solution, bounds, step_size):
    neighborhood = []
    for x in solution:
        min_x = max(bounds[0], x - step_size)
        max_x = min(bounds[1], x + step_size)
        new_x = np.random.uniform(min_x, max_x)
        neighborhood.append([new_x])  # Return each dimension as a list
    return neighborhood

def tabu_search(objective_func, bounds, dimensions, max_iterations, tabu_size, step_size):
    current_solution = generate_initial_solution(bounds, dimensions)
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
bounds = [-600, 600]
dimensions = 10
max_iterations = 100
tabu_size = 10
step_size = 10

best_solution = tabu_search(griewank, bounds, dimensions, max_iterations, tabu_size, step_size)
best_score = griewank(*best_solution)

print("Best solution:", best_solution)
print("Best score:", best_score)