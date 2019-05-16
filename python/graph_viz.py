from standard_vector import DynamicArray
from learned_standard_vector import LearnedDynamicArray
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import util
import cProfile, pstats, StringIO

if __name__ == '__main__':

    #Which graphs to plot?
    ''' 
    er=nx.erdos_renyi_graph(100,0.15)
    ws=nx.watts_strogatz_graph(10000,3,0.1)
    tree = nx.balanced_tree(2,10)
    complete = nx.complete_graph(100)
    ba=nx.barabasi_albert_graph(100,5)
    red=nx.random_lobster(100,0.9,0.9)    
    nx.draw(complete)
    plt.show()
    graph = complete
    '''
    graph=nx.complete_graph(10)
    nx.draw(graph)
    plt.show()

    graph=nx.complete_graph(100)

    learned_stack = LearnedDynamicArray()
    answer = util.dfs_iterative(graph, 1,learned_stack)


    n = plt.plot(learned_stack.history_n,label="number of elements in array")
    # c = plt.plot(learned_stack.history_capacity,label="capacity of the array")
    plt.xlabel("Timestemps")
    plt.ylabel("Size (in number of elements)")
    plt.legend()
    plt.show()


    n = np.array(learned_stack.history_n)/np.array(learned_stack.history_capacity).astype(np.float)
    plt.plot(n)
    plt.show()

    # n = np.array(learned_stack.history_capacity)-np.array(learned_stack.history_n).astype(np.float)
    # plt.plot(n/np.a)
    # plt.show()