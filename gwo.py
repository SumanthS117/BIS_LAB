import random
import math

# Objective functions
def sphere(x):
    return sum(val * val for val in x)

def sin_sq(x):
    return sum((math.sin(val))**2 for val in x)

# ------------------------- Grey Wolf Optimizer -------------------------

def GWO(func, dims=2, pack_size=30, iterations=100, bounds=(-10, 10)):
    """
    Rewritten Grey Wolf Optimizer (GWO)
    """

    lo, hi = bounds

    # Initialize wolf pack
    pack = [
        [random.uniform(lo, hi) for _ in range(dims)]
        for _ in range(pack_size)
    ]

    # Evaluate all wolves
    scores = [func(w) for w in pack]

    # Identify alpha, beta, delta (top 3)
    order = sorted(range(pack_size), key=lambda idx: scores[idx])
    alpha = pack[order[0]].copy()
    beta  = pack[order[1]].copy()
    delta = pack[order[2]].copy()

    fa, fb, fd = scores[order[0]], scores[order[1]], scores[order[2]]

    print(f"Initial α: {alpha} | f = {fa:.6f}")
    print(f"Initial β: {beta}  | f = {fb:.6f}")
    print(f"Initial δ: {delta} | f = {fd:.6f}\n")

    # Main loop
    for t in range(iterations):

        # a decreases linearly
        a = 2 * (1 - t / iterations)

        for i in range(pack_size):
            new_pos = []

            for d in range(dims):

                # Random coefficients for each leader
                r1a, r2a = random.random(), random.random()
                r1b, r2b = random.random(), random.random()
                r1d, r2d = random.random(), random.random()

                A1 = a * (2*r1a - 1)
                A2 = a * (2*r1b - 1)
                A3 = a * (2*r1d - 1)

                C1, C2, C3 = 2*r2a, 2*r2b, 2*r2d

                # Distances to leaders
                Da = abs(C1 * alpha[d] - pack[i][d])
                Db = abs(C2 * beta[d]  - pack[i][d])
                Dd = abs(C3 * delta[d] - pack[i][d])

                # Leader influence positions
                Xa = alpha[d] - A1 * Da
                Xb = beta[d]  - A2 * Db
                Xd = delta[d] - A3 * Dd

                # Update position
                x_new = (Xa + Xb + Xd) / 3
                x_new = max(lo, min(hi, x_new))  # Bound check
                new_pos.append(x_new)

            pack[i] = new_pos

        # Recalculate scores
        scores = [func(w) for w in pack]

        # Update alpha, beta, delta
        for i in range(pack_size):
            s = scores[i]

            if s < fa:
                fd, delta = fb, beta.copy()
                fb, beta  = fa, alpha.copy()
                fa, alpha = s, pack[i].copy()

            elif s < fb:
                fd, delta = fb, beta.copy()
                fb, beta = s, pack[i].copy()

            elif s < fd:
                fd, delta = s, pack[i].copy()

        if (t+1) % 20 == 0:
            print(f"Iter {t+1}: α = {alpha}, f = {fa:.6f}")

    return alpha, fa


# ------------------------------ Run ------------------------------

if __name__ == "__main__":
    print("="*70)
    print("                  Modified Grey Wolf Optimizer")
    print("="*70)

    best, value = GWO(
        func=sin_sq,
        dims=2,
        pack_size=30,
        iterations=100,
        bounds=(-math.pi, math.pi)
    )

    print("\n" + "="*70)
    print("                          RESULTS")
    print("="*70)
    print(f"Best vector   : {best}")
    print(f"Objective val : {value:.6f}")
    print("="*70)
    print("Hierarchy:")
    print("  α – top wolf")
    print("  β – second best")
    print("  δ – third best")
    print("  ω – remaining pack members")
    print("="*70)
