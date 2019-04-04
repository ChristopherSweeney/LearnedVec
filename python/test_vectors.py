from standard_vector import DynamicArray
from learned_standard_vector import LearnedDynamicArray
import numpy
from igraph import *
import cProfile
import matplotlib.pyplot as plt
import networkx as nx

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
        for neighbor in nx.all_neighbors(graph,vertex):
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

  
    # er=nx.erdos_renyi_graph(100,0.15)
    # ws=nx.watts_strogatz_graph(10000,3,0.1)
    # tree = nx.balanced_tree(2,10)
    complete = nx.complete_graph(100)
    # ba=nx.barabasi_albert_graph(100,5)
    # red=nx.random_lobster(100,0.9,0.9)    
    nx.draw(complete)
    plt.show()
    graph = complete
    print "----------------------------testing regular stack----------------------------"
    regular_stack = DynamicArray()
    cp = cProfile.Profile()
    cp.enable()
    answer = dfs_iterative(graph, 1,regular_stack)
    cp.disable()
    cp.print_stats()
    print "----------------------------testing learned stack----------------------------"
    learned_stack = LearnedDynamicArray()
    cp = cProfile.Profile()
    cp.enable()
    answer = dfs_iterative(graph, 1,learned_stack)
    cp.disable()
    cp.print_stats()
    plt.figure()
    plt.plot(learned_stack.history_n)
    plt.plot(learned_stack.history_capacity)
    plt.show()
    # print answer

