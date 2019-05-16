from standard_vector import DynamicArray
from learned_standard_vector import LearnedDynamicArray
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.svm import SVR
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

    train = []
    num_graphs = 10
    model = None
    stacks_cap = []
    ##################################Create Training Data####################################
    for i in range(num_graphs):
        graph = nx.random_lobster(100,0.9,0.9) 
        learned_stack = LearnedDynamicArray()
        answer = util.dfs_iterative(graph, 1,learned_stack)
        train.append(learned_stack.history_n)
        stacks_cap.append(learned_stack.history_capacity)

    ##################################Format Training Data####################################
    buckets = 20
    look_ahead_rate = 2
    train_x,train_y,choices = util.create_dataset(train,buckets=buckets,look_ahead_rate = look_ahead_rate)
    train_x = np.expand_dims(train_x ,2)
    train_y = np.expand_dims(train_y ,2)

    ##################################Train Model####################################
    lstm_units = 4
    model = util.build_lstm_model(lstm_units,buckets)
    model.fit(train_x,train_y, batch_size=4, epochs=5,verbose=1)

    ##################################Visualize Results####################################
    num_examples_to_view = 10
    for i in range(num_examples_to_view):
        predictions = model.predict(np.expand_dims(train_x[i] ,0))
        plt.figure()
        plt.plot(train[0])
        plt.plot(stacks_cap[0])
        index = choices[0][i]
        n = train[0][index]

        plt.scatter(np.arange(len(train_x[i]))*index/buckets,train_x[i]*n,color="g")
        plt.scatter(index, train_y[i]*n,color="r")
        plt.scatter(index, predictions*n,color="b")
        plt.show()

    ##################################Compare to model with no ML####################################
    graph = nx.random_lobster(100,0.9,0.9) 
    print "----------------------------testing regular stack----------------------------"
    gammas = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.]
    wasted_mem = []
    operations = []
    for gamma in gammas:
        regular_stack = LearnedDynamicArray(default_resize_up = gamma,default_resize_down=1./gamma,default_downsize_point=1./(gamma**2)) #with no model, this is just a regular dynamic array
    
        answer = util.dfs_iterative(graph, 1,regular_stack)
   
        reg_wasted_memory = np.array(regular_stack.history_capacity)- np.array(regular_stack.history_n)
        operations.append(regular_stack.operations)
        wasted_mem.append(np.sum(reg_wasted_memory))
        # plt.figure()
        # plt.plot(regular_stack.history_n)
        # plt.plot(regular_stack.history_capacity)
        # plt.show()

    print "----------------------------testing learned stack----------------------------"
    learned_stack = LearnedDynamicArray(model = model,buckets=buckets)
    answer = util.dfs_iterative(graph, 1,learned_stack)
    
    print "learned stack num total operations ", learned_stack.operations
    learned_wasted_memory = np.array(learned_stack.history_capacity)- np.array(learned_stack.history_n)
    m = np.sum(learned_wasted_memory)
    o = learned_stack.operations
    fig, ax = plt.subplots()
    ax.scatter(wasted_mem,operations)
    for i, txt in enumerate(gammas):
        ax.annotate(txt, (wasted_mem[i], operations[i]))

    ax.set_xlabel("wasted memory")
    ax.set_ylabel("operations")
    # ax.set_xlim(0, 1e8)
    # ax.set_ylim(0, 1000000)
    ax.scatter(m,o)
    plt.show()
    plt.figure()
    plt.plot(learned_stack.history_n)
    plt.plot(learned_stack.history_capacity)
    plt.show()

    gamma = 1.5
    regular_stack = LearnedDynamicArray(default_resize_up = gamma,default_resize_down=1./gamma,default_downsize_point=1./(gamma**2)) #with no model, this is just a regular dynamic array
    
    answer = util.dfs_iterative(graph, 1,regular_stack)
    plt.figure()
    plt.plot(regular_stack.history_n)
    plt.plot(regular_stack.history_capacity)
    plt.show()


