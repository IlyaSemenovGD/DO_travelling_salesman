#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import numpy as np
from scipy.spatial import distance
import random
from multiprocessing import Pool
from solver_parallel import simulated_annealing_parallel
import matplotlib.pyplot as plt
import time


        # try:
        #     dist_diff = math.exp((cur_distance - new_distance) / T)
        # except OverflowError:
        #     dist_diff = float('inf')        


Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


# Function to find longest edges in a tour
def find_longest_edges(tour, dist_matrix):
    longest_edges = []
    for i in range(len(tour)):
        for j in range(i + 2, len(tour)):
            edge_length = dist_matrix[tour[i], tour[j]]
            longest_edges.append((edge_length, i, j))
    longest_edges.sort(reverse=True)  # Sort by edge length
    return longest_edges


# Function to calculate total tour distance
def total_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1)) + dist_matrix[tour[-1], tour[0]]


# 2-opt local search procedure
def two_opt(tour, dist_matrix):
    best_tour = tour
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 2, len(tour) - 1):
                if dist_matrix[tour[i], tour[i+1]] + dist_matrix[tour[j], tour[j+1]] > dist_matrix[tour[i], tour[j]] + dist_matrix[tour[i+1], tour[j+1]]:
                    best_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                    improved = True
        tour = best_tour
    return best_tour


# Simulated annealing function with Metropolis heuristic, tabu search, and local search refinement
def simulated_annealing_metropolis_tabu_local_search(points, T=100, alpha=0.99, stopping_temp=1, stopping_iter=1000, tabu_size=5, runtime_limit=1800):
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

    # Initialize runtime tracking
    start_time = time.time()

    while (T > stopping_temp and iteration < stopping_iter) and (time.time() - start_time < runtime_limit):
        # Find longest edges in the current tour
        longest_edges = find_longest_edges(cur_tour, dist_matrix)

        # Calculate probabilities for selecting edges
        probabilities = np.array([edge[0] for edge in longest_edges])
        probabilities /= np.sum(probabilities)  # Normalize probabilities to sum up to 1
        
        # Select edge based on weighted probability
        selected_index = np.random.choice(len(longest_edges), p=probabilities)
        i, j = longest_edges[selected_index][1], longest_edges[selected_index][2]
        
        # Swap the endpoints of the selected edge
        new_tour = cur_tour[:i] + list(reversed(cur_tour[i:j+1])) + cur_tour[j+1:]

        # Apply 2-opt local search refinement
        new_tour = two_opt(new_tour, dist_matrix)

        # Calculate the new distance
        new_distance = total_distance(new_tour, dist_matrix)

        try:
            dist_diff = math.exp((cur_distance - new_distance) / T)
        except OverflowError:
            dist_diff = float('inf')        

        # Metropolis acceptance criterion
        if new_distance < cur_distance or random.random() < dist_diff:
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
        # print(f'{iteration}, {stopping_iter}')

    return best_tour, best_distance


def greedy_algorithm(points):
    num_points = len(points)
    center = Point(x=sum(p.x for p in points) / num_points, y=sum(p.y for p in points) / num_points)

    # Calculate distances from the center
    distances_from_center = [length(center, p) for p in points]

    # Find the farthest point from the center
    farthest_point_index = np.argmax(distances_from_center)

    # Greedy algorithm starting from the farthest point from the center
    unvisited = set(range(num_points))
    tour = [farthest_point_index]
    unvisited.remove(farthest_point_index)
    while unvisited:
        last_point = points[tour[-1]]
        nearest_neighbor_index = min(unvisited, key=lambda i: length(last_point, points[i]))
        tour.append(nearest_neighbor_index)
        unvisited.remove(nearest_neighbor_index)
        if len(unvisited) % 100 == 0:
            print(len(unvisited))

    return tour, total_distance(tour, distance.cdist(points, points, 'euclidean'))


def plot_points(points, tour):

    # Extract x and y coordinates from the list of tuples
    x_coords = [points[i][0] for i in tour]
    y_coords = [points[i][1] for i in tour]

    # Plot the points
    plt.scatter(x_coords, y_coords, color='blue')

    # Connect the points in the order of the tour
    for i in range(len(tour) - 1):
        plt.plot([points[tour[i]][0], points[tour[i+1]][0]], [points[tour[i]][1], points[tour[i+1]][1]], color='green')
    # Connect the last point to the first one
    plt.plot([points[tour[-1]][0], points[tour[0]][0]], [points[tour[-1]][1], points[tour[0]][1]], color='green')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Plot of Points with Tour')
    plt.grid(True)
    plt.savefig('plot.png')


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file

    hyperparams = {
        'T': 800,
        'alpha': 0.99,
        'stopping_temp': 0.001,
        'stopping_iter': 4000,
        'tabu_size': 15
    }

    if len(points) < 100:
        hyperparams['stopping_iter'] = 50000
    elif len(points) < 200:
        hyperparams['stopping_iter'] = 15000
    elif len(points) > 300:
        hyperparams['stopping_iter'] = 1000
    elif len(points) > 1500:
        hyperparams['stopping_iter'] = 50

    if len(points) < 1000:
        best_tour, best_distance = simulated_annealing_metropolis_tabu_local_search(points, 
                                                                        T=hyperparams['T'],
                                                                        alpha=hyperparams['alpha'],
                                                                        stopping_temp=hyperparams['stopping_temp'],
                                                                        stopping_iter=hyperparams['stopping_iter'],
                                                                        tabu_size=hyperparams['tabu_size'],
                                                                        runtime_limit=1800)  # runtime limit in seconds
    else:
        best_tour, best_distance = greedy_algorithm(points)

    print("Best tour:", best_tour)
    print("Best distance:", best_distance)
    plot_points(points, best_tour)

    solution = best_tour

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        solve_it(input_data)
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

