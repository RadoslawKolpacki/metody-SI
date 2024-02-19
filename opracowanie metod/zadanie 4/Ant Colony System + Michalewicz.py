import numpy as np

def michalewicz(args, m=10):
    x = args[0]
    y = args[1]
    term1 = np.sin(x) * (np.sin((x**2) / np.pi))**(2 * m)
    term2 = np.sin(y) * (np.sin((2 * y**2) / np.pi))**(2 * m)
    return -(term1 + term2)

class Ant:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], size=len(bounds))
        self.score = michalewicz(self.position)

class AntColonySystem:
    def __init__(self, objective_func, bounds, num_ants, num_iterations, alpha=1.0, beta=2.0, rho=0.5, q0=0.8):
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Pheromone factor
        self.beta = beta  # Heuristic factor
        self.rho = rho  # Pheromone evaporation rate
        self.q0 = q0  # Probability of selecting the best action (exploitation)
        self.best_solution = None
        self.best_score = float('inf')

    def optimize(self):
        num_dimensions = len(self.bounds)
        pheromone = np.ones((num_dimensions,))
        ants = [Ant(self.bounds) for _ in range(self.num_ants)]

        for iteration in range(self.num_iterations):
            for ant in ants:
                self.update_pheromone(ant, pheromone)
                self.local_search(ant)

            # Update global best solution
            for ant in ants:
                if ant.score < self.best_score:
                    self.best_solution = ant.position
                    self.best_score = ant.score

            self.global_update_pheromone(pheromone)

        return self.best_solution

    def update_pheromone(self, ant, pheromone):
        num_dimensions = len(self.bounds)
        delta_pheromone = np.zeros((num_dimensions,))
        for i in range(num_dimensions):
            delta_pheromone[i] = 1 / ant.score
        pheromone += delta_pheromone

    def local_search(self, ant):
        num_dimensions = len(self.bounds)
        original_position = ant.position.copy()
        original_score = ant.score
        for i in range(num_dimensions):
            new_position = np.random.normal(original_position[i], 0.1)
            new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
            ant.position[i] = new_position
            ant.score = self.objective_func(ant.position)
            if ant.score >= original_score:
                ant.position[i] = original_position[i]
                ant.score = original_score

    def global_update_pheromone(self, pheromone):
        pheromone *= (1 - self.rho)

    def select_action(self, ant, pheromone):
        probabilities = self.calculate_probabilities(ant, pheromone)
        if np.random.uniform() < self.q0:
            # Exploitation: Select the best action
            return np.argmax(probabilities)
        else:
            # Exploration: Select an action based on probabilities
            return np.random.choice(range(len(probabilities)), p=probabilities)

    def calculate_probabilities(self, ant, pheromone):
        num_dimensions = len(self.bounds)
        probabilities = np.zeros((num_dimensions,))
        for i in range(num_dimensions):
            if ant.position[i] > 0:
                probabilities[i] = (pheromone[i] ** self.alpha) * ((1 / ant.position[i]) ** self.beta)
            else:
                probabilities[i] = (pheromone[i] ** self.alpha) * ((1 / abs(ant.position[i])) ** self.beta)
        probabilities /= np.sum(probabilities)
        return probabilities

# Example usage:
bounds = [0, np.pi]  # Bounds for each variable
num_ants = 20
num_iterations = 100
alpha = 1.0
beta = 2.0
rho = 0.5
q0 = 0.8

acs = AntColonySystem(michalewicz, bounds, num_ants, num_iterations, alpha, beta, rho, q0)
best_solution = acs.optimize()
best_score = michalewicz(best_solution)

print("Best solution:", best_solution)
print("Best score:", best_score)
