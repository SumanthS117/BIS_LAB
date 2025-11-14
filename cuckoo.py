import random
import math

# Example objective: Sphere function
def cost_function(vector):
    return sum(v*v for v in vector)

# Lévy flight generator
def levy_step(dimensions):
    b = 1.5
    gamma_term = math.gamma(1+b) * math.sin(math.pi*b/2)
    denom = math.gamma((1+b)/2) * b * 2**((b-1)/2)
    sigma = (gamma_term / denom)**(1/b)

    numerator = random.gauss(0, sigma)
    denominator = abs(random.gauss(0, 1))**(1/b)
    step_base = numerator / denominator

    return [step_base * random.gauss(0, 1) for _ in range(dimensions)]

# Cuckoo Search rewritten uniquely
def run_cuckoo_search(f,
                      dims=2,
                      nests_count=25,
                      discovery_rate=0.25,
                      iterations=100,
                      domain=(-10, 10)):

    low, high = domain

    # Initial population
    population = [[random.uniform(low, high) for _ in range(dims)]
                  for _ in range(nests_count)]
    scores = [f(p) for p in population]

    # Track best nest
    best_idx = min(range(nests_count), key=lambda i: scores[i])
    best_pos = population[best_idx][:]
    best_score = scores[best_idx]

    print(f"Initial: x={best_pos}, f(x)={best_score:.6f}\n")

    for t in range(iterations):

        # Lévy updates
        for i in range(nests_count):
            jump = levy_step(dims)
            trial = [population[i][d] + jump[d] for d in range(dims)]

            # Bound control
            trial = [low if x < low else high if x > high else x for x in trial]

            score_trial = f(trial)
            rand_idx = random.randrange(nests_count)

            # Replace if better
            if score_trial < scores[rand_idx]:
                population[rand_idx] = trial
                scores[rand_idx] = score_trial

        # Abandon some nests
        abandon_num = int(discovery_rate * nests_count)

        # Indices sorted worst → best
        worst_to_best = sorted(range(nests_count), key=lambda k: scores[k], reverse=True)

        for id_bad in worst_to_best[:abandon_num]:
            population[id_bad] = [random.uniform(low, high) for _ in range(dims)]
            scores[id_bad] = f(population[id_bad])

        # Update global best
        new_best_idx = min(range(nests_count), key=lambda i: scores[i])
        if scores[new_best_idx] < best_score:
            best_score = scores[new_best_idx]
            best_pos = population[new_best_idx][:]

        if (t+1) % 20 == 0:
            print(f"Iter {t+1}: x={best_pos}, f(x)={best_score:.6f}")

    return best_pos, best_score

# Run algorithm
if __name__ == "__main__":
    print("=== Modified Cuckoo Search ===\n")

    position, value = run_cuckoo_search(
        f=cost_function,
        dims=2,
        nests_count=25,
        discovery_rate=0.25,
        iterations=100,
        domain=(-10, 10)
    )

    print("\n=== RESULT ===")
    print(f"Best vector : {position}")
    print(f"Fitness     : {value:.6f}")
