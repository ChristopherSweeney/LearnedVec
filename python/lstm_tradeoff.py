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
        graph = nx.watts_strogatz_graph(1000,3,0.1)
        learned_stack = LearnedDynamicArray()
        answer = util.dfs_iterative(graph, 1,learned_stack)
        train.append(learned_stack.history_n)
        stacks_cap.append(learned_stack.history_capacity)    

    ##################################Train Model####################################
    l= np.linspace(1.5,2.5,10)
    models = []
    for i in l:
        print i, " look_ahead_rate"
        buckets = 20
        look_ahead_rate = i
        train_x,train_y,choices = util.create_dataset(train,buckets=buckets,look_ahead_rate = look_ahead_rate)
        train_x = np.expand_dims(train_x ,2)
        train_y = np.expand_dims(train_y ,2)
        lstm_units = 4
        model = util.build_lstm_model(lstm_units,buckets)
        model.fit(train_x,train_y, batch_size=4, epochs=5,verbose=1)
        models.append(model)

    ##################################Compare to model with no ML####################################
    graph = nx.watts_strogatz_graph(3000,3,0.1)
    wasted_mem=[]
    operations = []

    for i in models:
        learned_stack = LearnedDynamicArray(model = model,buckets=buckets)
        answer = util.dfs_iterative(graph, 1,learned_stack)
        wasted_mem.append(np.array(learned_stack.history_capacity)- np.array(learned_stack.history_n))
        operations.append(learned_stack.operations)

    regular_stack = LearnedDynamicArray(default_resize_up = 2) #with no model, this is just a regular dynamic array
    answer = util.dfs_iterative(graph, 1,regular_stack)
    reg_wasted_memory = np.array(regular_stack.history_capacity)- np.array(regular_stack.history_n)
    fig, ax = plt.subplots()
    ax.scatter(wasted_mem,operations)
    for i, txt in enumerate(l):
        ax.annotate(txt, (wasted_mem[i], operations[i]))
    ax.set_xlabel("wasted memory")
    ax.set_ylabel("operations")
    # ax.set_xlim(0, 1e8)
    # ax.set_ylim(0, 1000000)
    ax.scatter(reg_wasted_memory,regular_stack.operations,color = 'g')
    plt.show()

        