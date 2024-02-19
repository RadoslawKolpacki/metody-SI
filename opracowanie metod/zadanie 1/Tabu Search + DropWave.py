import numpy as np


def drop_wave(x, y):
    numerator = 1 + np.cos(12 * np.sqrt(x ** 2 + y ** 2))
    denominator = 0.5 * (x ** 2 + y ** 2) + 2
    value = -numerator / denominator
    return value


def tabu_search(initial_solution, tabu_list_size, max_iterations):
    current_solution = initial_solution
    best_solution = current_solution
    tabu_list = []

    for _ in range(max_iterations):
        neighbors = get_neighbors(current_solution)
        best_neighbor = None
        best_neighbor_value = float('-inf')

        for neighbor in neighbors:
            if neighbor not in tabu_list:
                neighbor_value = drop_wave(neighbor[0], neighbor[1])
                if neighbor_value > best_neighbor_value:
                    best_neighbor = neighbor
                    best_neighbor_value = neighbor_value

        if best_neighbor is None:
            break

        current_solution = best_neighbor
        if best_neighbor_value > drop_wave(best_solution[0], best_solution[1]):
            best_solution = best_neighbor

        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)

    return best_solution


def get_neighbors(solution):
    neighbors = []
    for _ in range(10):
        neighbor = solution + np.random.uniform(-1, 1, 2)
        neighbors.append(neighbor.tolist())
    return neighbors


# Example usage
initial_solution = np.array([0, 0])
tabu_list_size = 10
max_iterations = 100

best_solution = tabu_search(initial_solution, tabu_list_size, max_iterations)
print("Best solution:", best_solution)
print("Best value:", drop_wave(best_solution[0], best_solution[1]))