from standard_vector import DynamicArray
from learned_standard_vector import LearnedDynamicArray
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.svm import SVR
import util
import cProfile

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
    num_epochs = 10
    clf=None
    buckets = 20
    stacks_n = []
    stacks_cap = []
    for i in range(num_epochs):
        graph = nx.erdos_renyi_graph(100,0.15)
        # nx.draw(graph)
        # plt.show()
        learned_stack = LearnedDynamicArray()
        answer = util.dfs_iterative(graph, 1,learned_stack)
        train.append(learned_stack.history_n)
        stacks_n.append(learned_stack.history_n)
        stacks_cap.append(learned_stack.history_capacity)

        #regress on mulitplicitve factor
        #decay factor
        #order
    train_x,train_y,choices = util.create_dataset(train,buckets=buckets)
    # clf = SVR( C=1.0, epsilon=0.2)
    # clf.fit(train_x, train_y)
    train_x = np.expand_dims(train_x ,2)
    train_y = np.expand_dims(train_y ,2)
    print np.shape(train_x)
    clf = util.build_lstm_model(buckets)
    clf.fit(train_x,train_y, batch_size=4, epochs=5,verbose=1)

    for i in range(5):
        print train_x[i],train_y[i],choices[0][i]
        predictions = clf.predict(np.expand_dims(train_x[i] ,0))
        plt.figure()
        plt.plot(stacks_n[0])
        plt.plot(stacks_cap[0])
        index = choices[0][i]
        print("index", index)
        print("learned_stack.history_n: ", len(learned_stack.history_n))
        n = stacks_n[0][index]

        plt.scatter(np.arange(len(train_x[i]))*index/buckets,train_x[i]*n,color="g")
        plt.scatter(index, train_y[i]*n,color="r")
        plt.scatter(index, predictions*n,color="b")
        plt.show()

    graph = nx.erdos_renyi_graph(100,0.15)
    print "----------------------------testing regular stack----------------------------"
    regular_stack = LearnedDynamicArray()
    cp = cProfile.Profile()
    cp.enable()
    answer = util.dfs_iterative(graph, 1,regular_stack)
    cp.disable()
    cp.print_stats()
    plt.figure()
    plt.plot(regular_stack.history_n)
    plt.plot(regular_stack.history_capacity)
    plt.show()
    print "----------------------------testing learned stack----------------------------"
    learned_stack = LearnedDynamicArray(model = clf,buckets=buckets)
    cp = cProfile.Profile()
    cp.enable()
    answer = util.dfs_iterative(graph, 1,learned_stack)
    cp.disable()
    cp.print_stats()
    plt.figure()
    plt.plot(learned_stack.history_n)
    plt.plot(learned_stack.history_capacity)
    plt.show()