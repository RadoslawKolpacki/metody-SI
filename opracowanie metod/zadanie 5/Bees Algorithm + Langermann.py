import numpy as np

def langermann(args):
    A = np.array([[3, 5], [5, 2], [2, 1], [1, 4]])
    c = np.array([1, 2, 5, 2])
    m = np.array([[3, 5], [5, 2], [2, 1], [1, 4]])
    result = 0
    for i in range(A.shape[0]):
        distance = np.sum((args - A[i]) ** 2)
        result += c[i] * np.exp(-distance / np.pi) * np.cos(np.pi * distance)
    return -result

class Bee:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], size=len(bounds))
        self.score = langermann(self.position)

class BeesAlgorithm:
    def __init__(self, objective_func, bounds, num_employed_bees, num_onlooker_bees, max_iterations):
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_employed_bees = num_employed_bees
        self.num_onlooker_bees = num_onlooker_bees
        self.max_iterations = max_iterations
        self.best_solution = None
        self.best_score = float('inf')

    def optimize(self):
        num_dimensions = len(self.bounds)
        employed_bees = [Bee(self.bounds) for _ in range(self.num_employed_bees)]

        for iteration in range(self.max_iterations):
            # Employed bees phase
            for bee in employed_bees:
                neighbor = self.explore_neighborhood(bee)
                if neighbor.score < bee.score:
                    bee.position = neighbor.position
                    bee.score = neighbor.score

            # Onlooker bees phase
            onlooker_bees = self.select_onlooker_bees(employed_bees)
            for bee in onlooker_bees:
                neighbor = self.explore_neighborhood(bee)
                if neighbor.score < bee.score:
                    bee.position = neighbor.position
                    bee.score = neighbor.score

            # Update global best solution
            for bee in employed_bees + onlooker_bees:
                if bee.score < self.best_score:
                    self.best_solution = bee.position
                    self.best_score = bee.score

        return self.best_solution

    def explore_neighborhood(self, bee):
        neighbor = Bee(self.bounds)
        neighbor.position = bee.position + np.random.uniform(-1, 1, size=len(bee.position))
        neighbor.position = np.clip(neighbor.position, self.bounds[0], self.bounds[1])
        neighbor.score = self.objective_func(neighbor.position)
        return neighbor

    def select_onlooker_bees(self, employed_bees):
        scores = np.array([bee.score for bee in employed_bees])
        non_negative_scores = scores - np.min(scores) + 1e-10  # Shift scores to make them non-negative
        probabilities = non_negative_scores / np.sum(non_negative_scores)
        selected_indices = np.random.choice(len(employed_bees), size=self.num_onlooker_bees, replace=False, p=probabilities)
        return [employed_bees[i] for i in selected_indices]

# Example usage:
bounds = [0, 10]  # Bounds for each variable
num_employed_bees = 50
num_onlooker_bees = 50
max_iterations = 100

bees_algo = BeesAlgorithm(langermann, bounds, num_employed_bees, num_onlooker_bees, max_iterations)
best_solution = bees_algo.optimize()
best_score = langermann(best_solution)

print("Best solution:", best_solution)
print("Best score:", best_score)