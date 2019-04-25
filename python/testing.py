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
        graph = nx.erdos_renyi_graph(100,0.15)
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
    num_examples_to_view = 1
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
    graph = nx.erdos_renyi_graph(1000,0.15)
    print "----------------------------testing regular stack----------------------------"
    regular_stack = LearnedDynamicArray() #with no model, this is just a regular dynamic array
    cp = cProfile.Profile()
    cp.enable()
    answer = util.dfs_iterative(graph, 1,regular_stack)
    cp.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(cp, stream=s).sort_stats(sortby)   
    ps.print_stats('resize|predict_horizon|pop|append')
    print s.getvalue() 

    reg_wasted_memory = np.array(regular_stack.history_capacity)- np.array(regular_stack.history_n)
    print "total wasted memory units: ", np.sum(reg_wasted_memory)
    print "average wasted memory units: ", np.mean(reg_wasted_memory)
    print "max wasted memory units: ", np.max(reg_wasted_memory)
    print "max memory capacity units: ", np.max(regular_stack.history_capacity)

    plt.figure()
    plt.plot(regular_stack.history_n)
    plt.plot(regular_stack.history_capacity)
    plt.show()
    print "----------------------------testing learned stack----------------------------"
    learned_stack = LearnedDynamicArray(model = model,buckets=buckets)
    cp = cProfile.Profile()
    cp.enable()
    answer = util.dfs_iterative(graph, 1,learned_stack)
    cp.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(cp, stream=s).sort_stats(sortby)   
    ps.print_stats('resize|predict_horizon|pop|append')
    print s.getvalue()

    learned_wasted_memory = np.array(learned_stack.history_capacity)- np.array(learned_stack.history_n)
    print "total wasted memory units: ", np.sum(learned_wasted_memory)
    print "average wasted memory units: ", np.mean(learned_wasted_memory)
    print "max wasted memory units: ", np.max(learned_wasted_memory)
    print "max memory capacity units: ", np.max(learned_stack.history_capacity)


    print "--------------------------comparison--------------------------"
    print "relative memory wasted" , np.sum(learned_wasted_memory)/float(np.sum(reg_wasted_memory))
    print "relative average wasted memory units: ", np.mean(learned_wasted_memory)/float(np.mean(reg_wasted_memory))
    print "relative max wasted memory units: ", np.max(learned_wasted_memory)/float(np.max(reg_wasted_memory))
    print "relative max memory capacity units: ", np.max(learned_stack.history_capacity)/float(np.max(regular_stack.history_capacity))
    plt.figure()
    plt.plot(learned_stack.history_n)
    plt.plot(learned_stack.history_capacity)
    plt.show()