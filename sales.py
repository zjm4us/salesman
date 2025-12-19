import numpy as np
import random
import math
import matplotlib.pyplot as plt
import argparse
from numba import njit, prange
from multiprocessing import Pool
import time

# Step 1: Read city coordinates from a .dat file
def read_coordinates(filename):
    """
    Reads city coordinates from a .dat file.
    Only reads the first two columns (longitude, latitude).
    Converts them to (latitude, longitude) format for calculations.
    """
    data = np.loadtxt(filename, skiprows=1, usecols=(0, 1))
    coords = data[:, [1, 0]]  # Convert to (lat, lon)
    return coords

# Step 2: Build distance matrix using Haversine formula
@njit(parallel=True)
def haversine_distance_matrix(coords):
    """
    Computes the full pairwise distance matrix between cities using
    the Haversine formula for great-circle distances on Earth.
    Uses Numba's JIT compilation for speed.
    """
    R = 6371.0  # Earth radius in km
    n = coords.shape[0]
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for i in prange(n):
        for j in range(n):
            dlat = lat[j] - lat[i]
            dlon = lon[j] - lon[i]
            a = math.sin(dlat / 2)**2 + math.cos(lat[i]) * math.cos(lat[j]) * math.sin(dlon / 2)**2
            c = 2 * math.asin(math.sqrt(a))
            dist_matrix[i, j] = R * c
    return dist_matrix

# Step 3: Compute total distance of a tour
@njit
def total_distance(tour, dist_matrix):
    """
    Computes the total length of a tour (round trip).
    """
    n = len(tour)
    dist = 0.0
    for i in range(n):
        dist += dist_matrix[tour[i], tour[(i + 1) % n]]  # wrap around for return to start
    return dist

# Nearest Neighbor heuristic
def nearest_neighbor(dist_matrix, start=0):
    """
    Quickly builds a tour using the nearest neighbor heuristic.
    Not guaranteed to be optimal, but gives a good benchmark.
    """
    n = dist_matrix.shape[0]
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    current = start

    while unvisited:
        next_city = min(unvisited, key=lambda city: dist_matrix[current, city])
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return np.array(tour)

# Step 4: Simulated Annealing worker
def simulated_annealing_worker(args):
    """
    Performs simulated annealing on a copy of the cities.
    Returns the best tour, its distance, and the annealing schedule.
    This function is designed to be run in parallel.
    """
    coords, dist_matrix, initial_temp, cooling_rate, max_iter = args
    n = len(coords)

    # Start with a random initial tour
    current_tour = np.arange(n)
    np.random.shuffle(current_tour)
    current_distance = total_distance(current_tour, dist_matrix)
    best_tour = current_tour.copy()
    best_distance = current_distance
    temp = initial_temp

    temperature_history = []
    distance_history = []

    for _ in range(max_iter):
        temperature_history.append(temp)
        distance_history.append(best_distance)

        # Select two indices at random for 2-opt swap
        a, b = np.random.choice(n, 2, replace=False)
        if a > b:
            a, b = b, a

        # Reverse the segment between a and b
        new_tour = current_tour.copy()
        new_tour[a:b+1] = new_tour[a:b+1][::-1]

        new_distance = total_distance(new_tour, dist_matrix)
        delta = new_distance - current_distance

        # Accept new tour if better, or with probability exp(-delta/T)
        if delta < 0 or np.random.rand() < math.exp(-delta / temp):
            current_tour = new_tour
            current_distance = new_distance
            if current_distance < best_distance:
                best_tour = current_tour.copy()
                best_distance = current_distance

        # Update temperature (cooling schedule)
        if temp > 100:
            temp *= cooling_rate
        else:
            temp *= 0.9995  # slower cooling at low temperatures

    return best_tour, best_distance, temperature_history, distance_history

# Step 5: Save optimized route to file
def save_route(filename, best_tour, coords):
    """
    Writes the optimized route to a .dat file compatible with routeplot.py
    """
    with open(filename, 'w') as f:
        f.write("#longitude\tlatitude\tcity\n")
        for idx in best_tour:
            lat, lon = coords[idx]
            f.write(f"{lon:.6f}\t{lat:.6f}\t{idx}\n")

# MAIN PROGRAM
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve TSP using Simulated Annealing with Numba and multiprocessing.")
    parser.add_argument("input_file", help="Path to .dat file with city coordinates")
    parser.add_argument("output_file", help="Path to save optimized route")
    parser.add_argument("--runs", type=int, default=10, help="Number of parallel SA runs")
    parser.add_argument("--initial_temp", type=float, default=200000, help="Initial temperature")
    parser.add_argument("--cooling_rate", type=float, default=0.996, help="Cooling rate")
    parser.add_argument("--max_iter", type=int, default=4000000, help="Maximum iterations per run")
    args = parser.parse_args()

    # Read coordinates and compute distance matrix
    coords = read_coordinates(args.input_file)
    dist_matrix = haversine_distance_matrix(coords)

    # NEW: Original route distance (file order)
    original_tour = np.arange(len(coords))
    original_distance = total_distance(original_tour, dist_matrix)

    # Compute nearest neighbor distance as benchmark
    nearest_tour = nearest_neighbor(dist_matrix, start=0)
    nearest_distance = total_distance(nearest_tour, dist_matrix)

    # Prepare arguments for parallel SA runs
    run_args = [(coords, dist_matrix, args.initial_temp, args.cooling_rate, args.max_iter)
                for _ in range(args.runs)]

    # Run simulated annealing in parallel
    start_time = time.perf_counter()
    with Pool(processes=args.runs) as pool:
        results = pool.map(simulated_annealing_worker, run_args)
    elapsed_time = time.perf_counter() - start_time

    # Pick the best run among all parallel runs
    best_run = min(results, key=lambda x: x[1])
    best_tour, best_distance, temperature_history, distance_history = best_run

    # Save optimized route to file
    save_route(args.output_file, best_tour, coords)

    # REQUIRED CONSOLE OUTPUT
    print(f"Original length (km): {original_distance:.2f}")
    print(f"Nearest neighbor (km): {nearest_distance:.2f}")
    print(f"Simulated annealing (km): {best_distance:.2f}")
    print(f"Time (s): {elapsed_time:.2f}")

    # Plot annealing schedule (distance vs temperature)
    plt.figure(figsize=(8, 5))
    plt.plot(temperature_history, distance_history, color='blue')
    plt.xlabel("Temperature")
    plt.xscale('log')
    plt.ylabel("Best Distance So Far (km)")
    plt.title("Annealing Schedule")
    plt.grid(True)

    # NEW: robust plot name extraction (cities150 → an150.png)
    tag = ''.join(filter(str.isdigit, args.input_file))
    if tag == "":
        tag = "23"
    plot_name = f"an{tag}.png"

    plt.savefig(plot_name)
    print(f"Annealing schedule saved as: {plot_name}")
    plt.show()

