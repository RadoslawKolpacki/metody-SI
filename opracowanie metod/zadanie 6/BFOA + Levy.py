import numpy as np

def levy(args):
    x = args[0]
    y = args[1]
    term1 = np.sin(3 * np.pi * x)**2
    term2 = (x - 1)**2 * (1 + np.sin(3 * np.pi * y)**2)
    term3 = (y - 1)**2 * (1 + np.sin(2 * np.pi * y)**2)
    return term1 + term2 + term3

class BacterialForagingOptimization:
    def __init__(self, objective_func, bounds, num_bacteria, num_iterations, chemotactic_step_size, swim_length, tumble_rate):
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_bacteria = num_bacteria
        self.num_iterations = num_iterations
        self.chemotactic_step_size = chemotactic_step_size
        self.swim_length = swim_length
        self.tumble_rate = tumble_rate
        self.best_solution = None
        self.best_score = float('inf')

    def optimize(self):
        num_dimensions = len(self.bounds)
        bacteria = self.initialize_bacteria()

        for iteration in range(self.num_iterations):
            for bact in bacteria:
                self.chemotaxis(bact)
                self.swim(bact)
                self.tumble(bact)

            # Update global best solution
            for bact in bacteria:
                if bact['score'] < self.best_score:
                    self.best_solution = bact['position']
                    self.best_score = bact['score']

        return self.best_solution

    def initialize_bacteria(self):
        bacteria = []
        for _ in range(self.num_bacteria):
            position = np.random.uniform(self.bounds[0], self.bounds[1], size=len(self.bounds))
            score = self.objective_func(position)
            bacteria.append({'position': position, 'score': score})
        return bacteria

    def chemotaxis(self, bact):
        num_dimensions = len(self.bounds)
        direction = np.random.uniform(-1, 1, size=num_dimensions)
        bact['position'] += self.chemotactic_step_size * direction
        bact['position'] = np.clip(bact['position'], self.bounds[0], self.bounds[1])
        bact['score'] = self.objective_func(bact['position'])

    def swim(self, bact):
        num_dimensions = len(self.bounds)
        swim_length = np.random.uniform(0, self.swim_length)
        direction = np.random.uniform(-1, 1, size=num_dimensions)
        bact['position'] += swim_length * direction
        bact['position'] = np.clip(bact['position'], self.bounds[0], self.bounds[1])
        bact['score'] = self.objective_func(bact['position'])

    def tumble(self, bact):
        if np.random.uniform() < self.tumble_rate:
            bact['position'] = np.random.uniform(self.bounds[0], self.bounds[1], size=len(self.bounds))
            bact['score'] = self.objective_func(bact['position'])

# Example usage:
bounds = [-10, 10]  # Bounds for each variable
num_bacteria = 50
num_iterations = 100
chemotactic_step_size = 0.1
swim_length = 0.2
tumble_rate = 0.1

bfoa = BacterialForagingOptimization(levy, bounds, num_bacteria, num_iterations, chemotactic_step_size, swim_length, tumble_rate)
best_solution = bfoa.optimize()
best_score = levy(best_solution)

print("Best solution:", best_solution)
print("Best score:", best_score)
