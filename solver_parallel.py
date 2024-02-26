import numpy as np
from scipy.spatial import distance
import random
import math
from multiprocessing import Pool, freeze_support

# Function to calculate total tour distance
def total_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1)) + dist_matrix[tour[-1], tour[0]]

# Function to find longest edges in a tour
def find_longest_edges(tour, dist_matrix):
    longest_edges = []
    for i in range(len(tour)):
        for j in range(i + 2, len(tour)):
            edge_length = dist_matrix[tour[i], tour[j]]
            longest_edges.append((edge_length, i, j))
    longest_edges.sort(reverse=True)  # Sort by edge length
    return longest_edges

# Function to compute distance between two points
def compute_distance(point1, point2):
    return distance.euclidean(point1, point2)

# Function to perform edge swap and calculate new distance
def perform_edge_swap(args):
    tour, dist_matrix, i, j = args
    new_tour = tour[:i] + list(reversed(tour[i:j+1])) + tour[j+1:]
    new_distance = total_distance(new_tour, dist_matrix)
    return new_tour, new_distance

# Simulated annealing function with parallelization
def simulated_annealing_parallel(points, T=1000, alpha=0.99, stopping_temp=0.0001, stopping_iter=1000, tabu_size=5, num_processes=None):
    num_points = len(points)
    # Create distance matrix
    dist_matrix = distance.cdist(points, points, 'euclidean')

    # Initialize tour
    cur_tour = list(range(num_points))
    best_tour = cur_tour
    np.random.shuffle(cur_tour)
    best_distance = total_distance(cur_tour, dist_matrix)
    cur_distance = best_distance

    # Initialize iteration counter
    iteration = 0

    # Initialize tabu list
    tabu_list = []

    # Set the number of processes for parallelization
    if num_processes is None:
        num_processes = min(4, num_points)  # Default to a reasonable number of processes

    while T > stopping_temp and iteration < stopping_iter:
        # Find longest edges in the current tour
        longest_edges = find_longest_edges(cur_tour, dist_matrix)

        # Calculate probabilities for selecting edges
        probabilities = np.array([edge[0] for edge in longest_edges])
        probabilities /= np.sum(probabilities)  # Normalize probabilities to sum up to 1
        
        # Select edge based on weighted probability
        selected_index = np.random.choice(len(longest_edges), p=probabilities)
        i, j = longest_edges[selected_index][1], longest_edges[selected_index][2]
        
        # Swap the endpoints of the selected edge using parallel computation
        pool = Pool(processes=num_processes)
        args_list = [(cur_tour, dist_matrix, i, j)] * num_processes
        results = pool.map(perform_edge_swap, args_list)
        pool.close()
        pool.join()

        # Choose the best solution among the parallel results
        new_tour, new_distance = min(results, key=lambda x: x[1])

        # Metropolis acceptance criterion
        if (new_distance < cur_distance) or (random.random() < math.exp((cur_distance - new_distance) / T)):
            cur_tour = new_tour
            cur_distance = new_distance

            # Update the best tour if necessary
            if cur_distance < best_distance:
                best_tour = cur_tour
                best_distance = cur_distance

        # Add the reversed edge to the tabu list
        tabu_list.append((i, j))
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        # Update the temperature and iteration count
        T *= alpha
        iteration += 1
        print(f'Iteration {iteration}')

    return best_tour, best_distance

if __name__ == '__main__':
    freeze_support()
    # Example usage:
    points = [(1, 2), (3, 4), (5, 6), (7, 8)]
    best_tour, best_distance = simulated_annealing_parallel(points)
    print("Best tour:", best_tour)
    print("Best distance:", best_distance)
