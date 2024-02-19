import numpy as np

def zakharov(*args):
    term1 = sum(args[i]**2 for i in range(len(args)))
    term2 = sum(0.5*(i+1)*args[i] for i in range(len(args)))
    return term1 + term2**2 + term2**4

class Particle:
    def __init__(self, bounds):
        self.position = [np.random.uniform(bounds[0], bounds[1]) for _ in range(len(bounds))]
        self.velocity = [np.random.uniform(-1, 1) for _ in range(len(bounds))]
        self.best_position = self.position
        self.best_score = zakharov(*self.position)

    def update_velocity(self, global_best_position, inertia_weight, cognitive_weight, social_weight):
        r1 = np.random.random()
        r2 = np.random.random()

        cognitive_component = cognitive_weight * r1 * (np.array(self.best_position) - np.array(self.position))
        social_component = social_weight * r2 * (np.array(global_best_position) - np.array(self.position))

        self.velocity = inertia_weight * np.array(self.velocity) + cognitive_component + social_component

    def update_position(self, bounds):
        self.position = np.clip(np.array(self.position) + np.array(self.velocity), bounds[0], bounds[1])
        score = zakharov(*self.position)
        if score < self.best_score:
            self.best_position = self.position
            self.best_score = score

def particle_swarm_optimization(objective_func, bounds, num_particles, max_iterations, inertia_weight, cognitive_weight, social_weight):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_score = float('inf')
    global_best_position = particles[0].position

    for _ in range(max_iterations):
        for particle in particles:
            particle.update_velocity(global_best_position, inertia_weight, cognitive_weight, social_weight)
            particle.update_position(bounds)
            score = objective_func(*particle.position)
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position

    return global_best_position

# Example usage:
bounds = [-5, 5]  # Bounds for each variable
num_particles = 50
max_iterations = 100
inertia_weight = 0.7
cognitive_weight = 1.4
social_weight = 1.4

best_solution = particle_swarm_optimization(zakharov, bounds, num_particles, max_iterations, inertia_weight, cognitive_weight, social_weight)
best_score = zakharov(*best_solution)

print("Best solution:", best_solution)
print("Best score:", best_score)
