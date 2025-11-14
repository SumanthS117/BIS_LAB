import numpy as np

# ------------------------------ INPUT ------------------------------

print("Enter an objective expression using numpy (variable: x)")
print("Example: np.sum(x**2)")
expr = input("f(x) = ")

# Build objective function
objective = eval(f"lambda x: {expr}")

dims = int(input("Number of dimensions: "))
low = float(input("Lower bound: "))
high = float(input("Upper bound: "))
rows = int(input("Grid rows: "))
cols = int(input("Grid cols: "))
iterations = int(input("Max iterations: "))

cell_count = rows * cols

# ------------------------------ INITIALIZATION ------------------------------

# Population grid of candidate vectors
grid = np.random.uniform(low, high, (rows, cols, dims))
scores = np.zeros((rows, cols))

best_val = float("inf")
best_vec = None

best_history = []
avg_history = []

# Neighbor offsets (Moore neighborhood)
offsets = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]

def neighbors_of(r, c):
    """Return wrapped neighbor indices."""
    return [((r+a) % rows, (c+b) % cols) for a, b in offsets]


print("\nOptimization started...")
print("-" * 50)

# ------------------------------ MAIN LOOP ------------------------------

for t in range(iterations):

    # Evaluate
    for r in range(rows):
        for c in range(cols):
            scores[r, c] = objective(grid[r, c])

            if scores[r, c] < best_val:
                best_val = scores[r, c]
                best_vec = grid[r, c].copy()

    avg_val = float(np.mean(scores))

    if t % 10 == 0:
        print(f"Iter {t:3d} | Best: {best_val:.6f} | Avg: {avg_val:.6f}")

    best_history.append(best_val)
    avg_history.append(avg_val)

    # Update grid for next iteration
    updated = grid.copy()

    for r in range(rows):
        for c in range(cols):

            # Pick best neighbor
            neigh = neighbors_of(r, c)
            best_n = None
            best_n_score = float("inf")

            for nr, nc in neigh:
                if scores[nr, nc] < best_n_score:
                    best_n_score = scores[nr, nc]
                    best_n = (nr, nc)

            # Move toward better neighbor
            if best_n_score < scores[r, c]:
                nr, nc = best_n
                step = 0.5
                direction = grid[nr, nc] - grid[r, c]
                updated[r, c] = grid[r, c] + step * direction

                # Clip to bounds
                updated[r, c] = np.clip(updated[r, c], low, high)

    grid = updated

# ------------------------------ RESULTS ------------------------------

print("-" * 50)
print("Optimization complete!")
print(f"Best vector : {best_vec}")
print(f"Best value  : {best_val:.10f}")
print("-" * 50)
