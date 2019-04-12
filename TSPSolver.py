#!/usr/bin/python3

import itertools
import heapq
from scipy.sparse.csgraph import minimum_spanning_tree as min_tree
from TSPClasses import *
import numpy as np
import time
from munkres import Munkres
from which_pyqt import PYQT_VER
from Christofides import christofides as cf


if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def greedy(self, time_allowance=60.0):
        # Greedy algorithm to find a first tour fast that is near the optimal
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        bssf = None
        start_time = time.time()
        start_node = 0
        # Loop until a tour is found or we time out
        while not foundTour and time.time() - start_time < time_allowance and start_node < ncities:
            badTour = False
            current_matrix = self.generateInitialMatrix()
            route = []
            # Start at the current start node, this will loop through every possible node as a start until a valid tour
            # is found or we time out
            current_node = start_node
            route.append(cities[current_node])
            current_matrix = blockCol(current_matrix, current_node)
            # We loop through enough times to create the length of a tour
            for i in range(ncities - 1):
                # From our current node grab the index for the smallest cost
                current_node = self.findMinIndex(current_matrix, current_node)
                # if our current_node is infinite then that means the lowest cost was infinite so this won't be a valid tour
                if current_node == np.inf:
                    badTour = True
                    break
                # append the node to the route and update the matrix so we don't revisit
                route.append(cities[current_node])
                current_matrix = blockCol(current_matrix, current_node)
            # create a TSPSolution based on our tour, if we had a bad tour or if the cost is infinite then it is not valid
            # so throw it out and try the next node as our starting position
            bssf = TSPSolution(route)
            if badTour:
                bssf.cost = np.inf
            if bssf.cost < np.inf:
                foundTour = True
                self._global_bssf = bssf
            start_node += 1
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else np.inf
        results['time'] = end_time - start_time
        results['count'] = None
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def branchAndBound(self, time_allowance=60.0):
        pass

    def generateInitialMatrixBranch(self):
        i = 0
        j = 0
        cities = self._scenario.getCities()
        ncities = len(cities)
        matrix = np.empty([ncities, ncities])
        for i in range(ncities):
            for j in range(i,ncities):
                matrix[i, j] = cities[i].costTo(cities[j])
        return matrix


    def fancy(self, time_allowance=60.0):
        results = {}
        start_time = time.time()
        initial_matrix = self.generateInitialMatrix()
        print("initial matrix:")
        print(time.time()-start_time)
        # print("{}\n".format(initial_matrix))
        min_tree = self.minTree(initial_matrix)
        print("min_tree:")
        print(time.time()-start_time)
        # print("{}\n".format(min_tree))
        odd_verts = self.getOddVerts(min_tree)
        print("oddverts:")
        print(time.time()-start_time)
        print("percent odd" + str(len(odd_verts) * 100 / initial_matrix.shape[0]))
        perfect = self.perfectMatchNetwork(odd_verts,initial_matrix,min_tree)
        perfectGreedy = self.perfectMatchGreedy(odd_verts, initial_matrix.copy(), min_tree)
        print("perfectGreedy:")
        print(time.time()-start_time)
        multigraph = self.multigraph(min_tree, perfect)
        self.convert_to_dir_graph(multigraph)
        num_edges = self.getEdges(multigraph)
        if len(self.getOddVerts(multigraph)) != 0:
            print("Uneven nodes!!!")
        print("multigraph:")
        print(time.time()-start_time)
        # print("{}\n".format(multigraph))
        # print(num_edges)
        euclidGraph = self.hierholzer(multigraph, num_edges)
        print("euclidian:")
        print(time.time()-start_time)
        # print(euclidGraph)
        tour, tracker = self.shortcut(euclidGraph)
        # print(tracker)
        christof_aprox = TSPSolution(tour)
        end_time = time.time()
        results['cost'] = christof_aprox.cost
        results['time'] = end_time - start_time
        results['count'] = None
        results['soln'] = christof_aprox
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    def generateInitialMatrix(self):
        i = 0
        j = 0
        cities = self._scenario.getCities()
        ncities = len(cities)
        matrix = np.empty([ncities, ncities])
        for i in range(ncities):
            for j in range(ncities):
                matrix[i, j] = cities[i].costTo(cities[j])
        return matrix

    def getOddVerts(self, matrix):
        odds = []
        for i in range(matrix.shape[0]):
            size = 0
            for j in range(matrix.shape[0]):
                if matrix[i, j] > 0:
                    size += 1
            for k in range(matrix.shape[0]):
                if matrix[k, i] > 0:
                    size += 1
            if (size % 2) != 0:
                odds.append(i)
        return odds

    def minTree(self, matrix):
        min_matrix = min_tree(matrix)
        min_matrix = min_matrix.toarray().astype(float)
        return min_matrix

    def findMinIndex(self, matrix, row):
        minIndex = np.inf
        min = np.inf
        for i in range(matrix.shape[1]):
            if matrix[row, i] < min:
                minIndex = i
                min = matrix[row, i]
        return minIndex

    # Source: https://github.com/sonph/pygraph/blob/master/pygraphalgo.py
    def hierholzer(self, graph, num_edges):
        # Convert undirected graph into a directed graph
        self.convert_to_dir_graph(graph)
        # Initialize variables
        start_vertex = 0
        circuit = [start_vertex]
        edges_visited = []
        current_node_index = 0
        # Loop through all vertices in the circuit and make sure they don't have any unvisited edges
        while len(edges_visited) < num_edges:
            # Initialize current path to be updated from following the edge to the next vertex
            curr_path = []
            self.search_new_vertex(
                graph, circuit[current_node_index], curr_path, edges_visited, start_vertex)
            current_node_index += 1
            insert_index = current_node_index
            # Add the new path to the current circuit
            for i in range(len(curr_path)):
                circuit.insert(insert_index, curr_path[i])
                insert_index += 1
            if current_node_index >= len(circuit):
                break

        return circuit

    def convert_to_dir_graph(self, graph):
        # Loop through every cell and make sure that its inverse cell is equal to it
        for i in range(len(graph)):
            for j in range(len(graph)):
                if graph[i, j] != np.inf and graph[i, j] != graph[j, i]:
                    graph[j, i] = graph[i, j]

    def search_new_vertex(self, graph, u, curr_path, edges_visited, starting_vertex):
        # Loop through all edges that connect to the current vertex (u)
        for v in range(graph.shape[0]):
            # If an edge exists and it hasn't been visited
            if graph[u][v] != np.inf and (u, v) not in edges_visited:
                # Mark as visited
                edges_visited.append((u, v))
                edges_visited.append((v, u))
                # Add it to the current path
                curr_path.append(v)
                # If we have completed the circuit, return; else, keep searching until the circuit is completed
                if v == starting_vertex:
                    return
                else:
                    self.search_new_vertex(
                        graph, v, curr_path, edges_visited, starting_vertex)


    def perfectMatchNetwork(self, vertices, matrix, min_matrix):
        # for i in range(matrix.shape[0]):
            # for j in range(matrix.shape[0]):
            #     if min_matrix[i][j] != 0 or min_matrix[j][i] != 0:
            #         matrix[i][j] = np.inf
        newmatrix = np.zeros(min_matrix.shape)
        bipartite_set = [set(i) for i in itertools.combinations(set(vertices), len(vertices) // 2)]
        bipartite_graphs = self.bipartite_Graph(matrix, bipartite_set, vertices)
        indexes = self.min_Munkres(matrix, bipartite_graphs)
        for pair in indexes:
            newmatrix[pair[0]][pair[1]] = matrix[pair[0]][pair[1]]
        return newmatrix

    def bipartite_Graph(self, M, bipartite_set, odd_vertices):
        """
        """
        bipartite_graphs = []
        vertex_sets = []
        for vertex_set1 in bipartite_set:
            vertex_set1 = list(sorted(vertex_set1))
            vertex_set2 = []
            for vertex in odd_vertices:
                if vertex not in vertex_set1:
                    vertex_set2.append(vertex)
            matrix = [[np.inf for j in range(len(vertex_set2))] for i in range(len(vertex_set1))]
            for i in range(len(vertex_set1)):
                for j in range(len(vertex_set2)):
                    if vertex_set1[i] < vertex_set2[j]:
                        matrix[i][j] = M[vertex_set1[i]][vertex_set2[j]]
                    else:
                        matrix[i][j] = M[vertex_set2[j]][vertex_set1[i]]
            bipartite_graphs.append(matrix)
            vertex_sets.append([vertex_set1, vertex_set2])
        return [bipartite_graphs, vertex_sets]

    def min_Munkres(self,M, bipartite_graphs):
        """Implements the Hungarian problem or the Assignment problem to
        find Minimum Cost Perfect Matching(MCPM).

        """
        m = Munkres()
        minimum = np.inf
        for index, bipartite_graph in enumerate(bipartite_graphs[0]):
            Munkres_indexes = m.compute(bipartite_graph)
            cost = self.Munkres_cost(Munkres_indexes, bipartite_graph)
            if cost < minimum:
                minimum = cost
                min_index = index
                min_Munkres_indexes = Munkres_indexes
        Munkres_indexes = [[] for i in range(len(min_Munkres_indexes))]
        for index, vertex_set in enumerate(min_Munkres_indexes):
            Munkres_indexes[index].append(bipartite_graphs[1][min_index][0][vertex_set[0]])
            Munkres_indexes[index].append(bipartite_graphs[1][min_index][1][vertex_set[1]])
        return Munkres_indexes

    def Munkres_cost(self, indexes, bipartite_graph):
        """Returns cost of the edges in Munkres_indexes

        """
        cost = 0
        for index in indexes:
            cost = cost + bipartite_graph[index[0]][index[1]]
        return cost

    def perfectMatchGreedy(self, vertices, matrix, minMatrix):
        newmatrix = np.zeros(matrix.shape)
        numvertices = len(vertices)
        # mark distances to all even degree vertexes as infinity
        for i in range(matrix.shape[0]):
            if i not in vertices:
                matrix[i] = math.inf
                for j in range(matrix.shape[1]):
                    matrix[j][i] = math.inf
        while len(vertices) != 0:
            # there should always be an even number of vertices
            if len(vertices) == 1:
                print("this should never happen")
            else:
                pos = np.argmin(matrix)
                cols = matrix.shape[0]
                # calculate location of smallest edge
                y = np.mod(pos, cols)
                x = pos // matrix.shape[0]
                # check if both vertices are in still in contention
                if x in vertices and y in vertices:
                    if minMatrix[x][y] == 0 and minMatrix[y][x] == 0 and matrix[x][y] != np.inf:
                        # print("adding match edge --> y (col) = {}, x (row) = {}".format(y, x))
                        # print("{}\n".format(matrix))
                        #when a position is found, remove the two vertices from the array
                        vertices.remove(x)
                        vertices.remove(y)
                        newmatrix[x][y] = matrix[x][y]
                    #once a position has been considered, mark it as infinity so that the next one can be found
                    matrix[x][y] = math.inf
                    matrix[y][x] = math.inf
                    if not vertices:
                        return newmatrix
                else:
                    matrix[x][y] = math.inf
                    matrix[y][x] = math.inf
                    continue
        return newmatrix

    def checkPerfect(self, matrix, numvertices):
        # get the minimum values of each column
        min = matrix.max(1)
        # if vertices // 2 edges have been added, it is a perfect match
        if np.count_nonzero(min) == numvertices // 2:
            return True
        else:
            return False

    def multigraph(self, matrix, perfectMatrix):
        newmatrix = matrix + perfectMatrix
        for i in range(newmatrix.shape[0]):
            for j in range(newmatrix.shape[0]):
                if newmatrix[i][j] == 0:
                    newmatrix[i][j] = np.inf
        return newmatrix

    def shortcut(self, circuit):
        # follow Eulerian circuit adding vertices on first encounter
        cities = self._scenario.getCities()
        Tour = []
        tracker = []
        for vert in circuit:
            if vert not in tracker:
                tracker.append(vert)
                Tour.append(cities[vert])

        return Tour, tracker

    def getEdges(self, matrix):
        toReturn = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                if matrix[i][j] != np.inf:
                    toReturn += 1
        return toReturn
