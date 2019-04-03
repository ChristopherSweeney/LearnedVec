from standard_vector import DynamicArray
from learned_standard_vector import LearnedDynamicArray
import numpy
from igraph import *
import cProfile
import matplotlib.pyplot as plt


# def generate_gaussian_queries_append(num):

# def generate_linear_queries_append(start_size,num_queries):
# 		return append()start_size,start_size+num_queries)

#taken from https://www.koderdojo.com/blog/depth-first-search-in-python-recursive-and-non-recursive-programming
def dfs_iterative(graph, start, stack):
    path =  []
    stack.append(start)
    while stack:
        vertex = stack.pop()
        if vertex in path:
            continue
        path.append(vertex)
        for neighbor in graph[vertex]:
            stack.append(neighbor)
    return path

if __name__ == '__main__':
    # arr = DynamicArray()
    # print("\ntesting appending\n")
    # for i in range(100):
    #     print(i)
    #     print(arr.append(i))
    #     print("capacity: ", arr.capacity)
    #     print("size: ", len(arr))

    # print("\ntesting popping\n")
    # for i in range(100):
    #     e = arr.pop()
    #     print(e)
    #     print("capacity: ", arr.capacity)
    #     print("size: ", len(arr))

    # adjacency_matrix = {1: [2, 3], 2: [4, 5],
    #                 3: [5], 4: [6], 5: [6],
    #                 6: [7], 7: []}

    #generate random graph
    #could also use code at https://networkx.github.io/documentation/networkx-1.10/reference/generators.html
    edges=1000
    prob_thresh=.70
    adj = numpy.random.rand(edges, edges)
    adj[adj > prob_thresh] = 1 # sets everything over 0.999 to 1
    adj[adj <= prob_thresh] = 0 # sets everything below to 0
    graph = Graph.Adjacency(list(adj))
    # print graph
    print "----------------------------testing regular stack----------------------------"
    regular_stack = DynamicArray()
    cp = cProfile.Profile()
    cp.enable()
    answer = dfs_iterative(graph.get_adjlist(), 1,regular_stack)
    cp.disable()
    cp.print_stats()
    print "----------------------------testing learned stack----------------------------"
    learned_stack = LearnedDynamicArray()
    cp = cProfile.Profile()
    cp.enable()
    answer = dfs_iterative(graph.get_adjlist(), 1,learned_stack)
    cp.disable()
    cp.print_stats()
    plt.figure()
    plt.plot(learned_stack.history_n)
    plt.plot(learned_stack.history_capacity)
    plt.show()
    # print answer
	#[1, 3, 5, 6, 7, 2, 4]

