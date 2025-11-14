import numpy as np
import random

# Step 1: Define the Problem - Create cities with random coordinates
def create_cities(num_cities):
    """Generate random city coordinates"""
    cities = []
    for i in range(num_cities):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        cities.append((x, y))
    return cities

# Calculate Euclidean distance between two cities
def calculate_distance(city1, city2):
    """Calculate distance between two cities"""
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Create distance matrix for all cities
def create_distance_matrix(cities):
    """Build a matrix of distances between all city pairs"""
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = calculate_distance(cities[i], cities[j])
    return dist_matrix

# Step 3: Construct Solutions - Select next city probabilistically
def select_next_city(current_city, visited, pheromones, distances, alpha, beta, n_cities):
    """Probabilistically select next city based on pheromone and distance"""
    unvisited = [i for i in range(n_cities) if i not in visited]

    if len(unvisited) == 1:
        return unvisited[0]

    probabilities = []
    for city in unvisited:
        pheromone = pheromones[current_city][city] ** alpha
        heuristic = (1.0 / distances[current_city][city]) ** beta
        probabilities.append(pheromone * heuristic)

    probabilities = np.array(probabilities)
    probabilities = probabilities / probabilities.sum()

    next_city = np.random.choice(unvisited, p=probabilities)
    return next_city

# Construct a complete tour for one ant
def construct_solution(n_cities, pheromones, distances, alpha, beta):
    """One ant constructs a complete tour"""
    route = [random.randint(0, n_cities - 1)]

    while len(route) < n_cities:
        current_city = route[-1]
        next_city = select_next_city(current_city, route, pheromones, distances, alpha, beta, n_cities)
        route.append(next_city)

    return route

# Calculate total distance of a route
def calculate_route_distance(route, distances):
    """Calculate total distance of a route"""
    distance = 0
    for i in range(len(route)):
        from_city = route[i]
        to_city = route[(i + 1) % len(route)]
        distance += distances[from_city][to_city]
    return distance

# Step 4: Update Pheromones
def update_pheromones(pheromones, all_routes, all_distances, rho, q):
    """Update pheromone trails based on ant solutions"""
    pheromones *= (1 - rho)

    for route, distance in zip(all_routes, all_distances):
        pheromone_deposit = q / distance
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            pheromones[from_city][to_city] += pheromone_deposit
            pheromones[to_city][from_city] += pheromone_deposit

    return pheromones

# Step 5 & 6: Main ACO algorithm
def solve_tsp_aco(cities, n_ants=20, n_iterations=100, alpha=1.0, beta=2.0, rho=0.5, q=100):
    """Main ACO algorithm to solve TSP"""
    print("Starting Ant Colony Optimization...")

    n_cities = len(cities)
    distances = create_distance_matrix(cities)
    pheromones = np.ones((n_cities, n_cities)) * 0.1

    best_route = None
    best_distance = float('inf')
    distance_history = []

    for iteration in range(n_iterations):
        all_routes = []
        all_distances = []

        for ant in range(n_ants):
            route = construct_solution(n_cities, pheromones, distances, alpha, beta)
            distance = calculate_route_distance(route, distances)
            all_routes.append(route)
            all_distances.append(distance)

            if distance < best_distance:
                best_distance = distance
                best_route = route

        pheromones = update_pheromones(pheromones, all_routes, all_distances, rho, q)
        distance_history.append(best_distance)

        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}, Best Distance: {best_distance:.2f}")

    print("\nOptimization Complete!")
    return best_route, best_distance, distance_history

# Main execution
if __name__ == "__main__":
    num_cities = 15
    cities = create_cities(num_cities)
    print(f"Created {num_cities} cities")

    best_route, best_distance, distance_history = solve_tsp_aco(
        cities=cities,
        n_ants=20,
        n_iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.5,
        q=100
    )

    print("\n" + "="*50)
    print("BEST SOLUTION FOUND")
    print("="*50)
    print(f"Route: {best_route}")
    print(f"Total Distance: {best_distance:.2f}")
    print("="*50)
