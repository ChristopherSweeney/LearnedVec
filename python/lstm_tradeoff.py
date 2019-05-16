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
    stacks_cap = []
    ##################################Create Training Data####################################
    for i in range(num_graphs):
        graph = nx.complete_graph(100)#nx.watts_strogatz_graph(1000,3,0.1)
        learned_stack = LearnedDynamicArray()
        answer = util.dfs_iterative(graph, 1,learned_stack)
        train.append(learned_stack.history_n)
        stacks_cap.append(learned_stack.history_capacity)    

    ##################################Train Model####################################
    l= np.around(np.linspace(1.5,3,10),1)
    models = []
    for i in l:
        print i, " look_ahead_rate"
        buckets = 20
        look_ahead_rate = i
        train_x,train_y,choices = util.create_dataset(train,buckets=buckets,look_ahead_rate = look_ahead_rate)
        train_x = np.expand_dims(train_x ,2)
        train_y = np.expand_dims(train_y ,2)
        lstm_units = 8
        model = util.build_lstm_model(lstm_units,buckets)
        history = model.fit(train_x,train_y, batch_size=16, epochs=10,verbose=1)
        while float(history.history['loss'][-1])>1:
            model = util.build_lstm_model(lstm_units,buckets)
            history = model.fit(train_x,train_y, batch_size=16, epochs=10,verbose=1)
        models.append(model)

    ##################################Compare to model with no ML####################################
    graph = nx.complete_graph(110) #nx.watts_strogatz_graph(3000,3,0.1)
    wasted_mem=[]
    operations = []

    for j,i in enumerate(models):
        learned_stack = LearnedDynamicArray(model = i,buckets=buckets)
        answer = util.dfs_iterative(graph, 1,learned_stack)
        wasted_mem.append(np.sum(np.array(learned_stack.history_capacity)- np.array(learned_stack.history_n)))
        operations.append(learned_stack.operations)
        print j
        plt.plot(learned_stack.history_n)
        plt.plot(learned_stack.history_capacity)
        plt.show()

    regular_stack = LearnedDynamicArray(default_resize_up = 2) #with no model, this is just a regular dynamic array
    answer = util.dfs_iterative(graph, 1,regular_stack)
    plt.plot(regular_stack.history_n)
    plt.plot(regular_stack.history_capacity)
    plt.show()


    

    reg_wasted_memory = np.sum(np.array(regular_stack.history_capacity)- np.array(regular_stack.history_n))

    fig, ax = plt.subplots()
    ax.scatter(wasted_mem,operations)
    for i, txt in enumerate(l):
        ax.annotate(txt, (wasted_mem[i], operations[i]))
    ax.set_xlabel("wasted memory")
    ax.set_ylabel("operations")
    # ax.set_xlim(0, 1e8)
    # ax.set_ylim(0, 1000000)


    gammas = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.]
    wasted_mem1 = []
    operations1 = []
    for gamma in gammas:
        regular_stack = LearnedDynamicArray(default_resize_up = gamma,default_resize_down=1./gamma,default_downsize_point=1./(gamma**2)) #with no model, this is just a regular dynamic array
    
        answer = util.dfs_iterative(graph, 1,regular_stack)
   
        reg_wasted_memory = np.array(regular_stack.history_capacity)- np.array(regular_stack.history_n)
        operations1.append(regular_stack.operations)
        wasted_mem1.append(np.sum(reg_wasted_memory))

    ax.scatter(wasted_mem1,operations1,color="y")
    for i, txt in enumerate(gammas):
        ax.annotate(txt, (wasted_mem1[i], operations1[i]))

    plt.show()

        