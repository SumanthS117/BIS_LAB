import random

def objective(x, y):
    return x**2 + y**2

num_particles = 10
num_iterations = 50
inertia = 0.5
personal_weight = 1.5
social_weight = 1.5

particle_positions = [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(num_particles)]
particle_velocities = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(num_particles)]
personal_best_positions = [list(pos) for pos in particle_positions]
personal_best_scores = [objective(pos[0], pos[1]) for pos in particle_positions]

best_score = min(personal_best_scores)
best_index = personal_best_scores.index(best_score)
global_best_position = list(personal_best_positions[best_index])

for iteration in range(num_iterations):
    for i in range(num_particles):
        for dim in range(2):
            r1 = random.random()
            r2 = random.random()
            personal_component = personal_weight * r1 * (personal_best_positions[i][dim] - particle_positions[i][dim])
            social_component = social_weight * r2 * (global_best_position[dim] - particle_positions[i][dim])
            particle_velocities[i][dim] = inertia * particle_velocities[i][dim] + personal_component + social_component

        particle_positions[i][0] += particle_velocities[i][0]
        particle_positions[i][1] += particle_velocities[i][1]

        score = objective(particle_positions[i][0], particle_positions[i][1])

        if score < personal_best_scores[i]:
            personal_best_positions[i] = list(particle_positions[i])
            personal_best_scores[i] = score
            if score < objective(global_best_position[0], global_best_position[1]):
                global_best_position = list(particle_positions[i])

    if iteration % 10 == 0:
        print("Iteration:", iteration, "Best Position:", global_best_position, "Score:", objective(global_best_position[0], global_best_position[1]))

print("Final Best Solution:", global_best_position)
