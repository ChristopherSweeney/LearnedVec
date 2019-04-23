from standard_vector import DynamicArray
from learned_standard_vector import LearnedDynamicArray
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.svm import SVR

if __name__ == '__main__':
    # # er=nx.erdos_renyi_graph(100,0.15)
    # # ws=nx.watts_strogatz_graph(10000,3,0.1)
    # # tree = nx.balanced_tree(2,10)
    # complete = nx.complete_graph(100)
    # # ba=nx.barabasi_albert_graph(100,5)
    # # red=nx.random_lobster(100,0.9,0.9)    
    # nx.draw(complete)
    # plt.show()
    # graph = complete
    train = []
    num_epochs = 5
    clf=None
    for i in range(num_epochs):
        graph = nx.complete_graph(100)
        # nx.draw(graph)
        # plt.show()
        learned_stack = LearnedDynamicArray()
        answer = dfs_iterative(graph, 1,learned_stack)
        train.append(learned_stack.history_n)
        #regress on mulitplicitve factor
        #decay factor
        #order
    train_x,train_y,choices = create_dataset(train)
    clf = SVR( C=1.0, epsilon=0.2)
    clf.fit(train_x, train_y)
    
    for i in range(5):
        predictions = clf.predict([train_x[i]])
        plt.figure()
        plt.plot(learned_stack.history_n)
        plt.plot(learned_stack.history_capacity)
        plt.scatter(np.arange(len(train_x[i]))*choices[0][i]/20,train_x[i],color="g")
        plt.scatter(choices[0][i], train_y[i],color="r")
        plt.scatter(choices[0][i], predictions,color="b")
        plt.show()

    print "----------------------------testing regular stack----------------------------"
    regular_stack = LearnedDynamicArray()
    cp = cProfile.Profile()
    cp.enable()
    answer = dfs_iterative(graph, 1,regular_stack)
    cp.disable()
    cp.print_stats()
    plt.figure()
    plt.plot(regular_stack.history_n)
    plt.plot(regular_stack.history_capacity)
    plt.show()
    print "----------------------------testing learned stack----------------------------"
    complete = nx.complete_graph(100)
    learned_stack = LearnedDynamicArray(model = clf,buckets=20)
    cp = cProfile.Profile()
    cp.enable()
    answer = dfs_iterative(graph, 1,learned_stack)
    cp.disable()
    cp.print_stats()
    plt.figure()
    plt.plot(learned_stack.history_n)
    plt.plot(learned_stack.history_capacity)
    plt.show()