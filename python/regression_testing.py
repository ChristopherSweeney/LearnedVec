from standard_vector import DynamicArray
from learned_standard_vector import LearnedDynamicArray
import numpy as np
from igraph import *
import cProfile
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import find_peaks

def create_dataset(data,buckets=10): #data [history_n], history_n :[1,2,3]
    data_y = []
    data_x = []
    for i,graph_hist in enumerate(data):
        print "ingesting graph "+str(i)
        x,y = convert_training_graph_history(graph_hist,buckets)
        data_y.extend(y)
        data_x.extend(x)
    return data_x, data_y

def convert_training_sample(graph_hist,sample,buckets):
    if sample<buckets: return None,None
    prev = graph_hist[:sample]
    bucket_size = int(len(prev) / buckets)
    print "sample ",sample
    print "bucket_size", bucket_size
    pad_size = bucket_size - int(len(prev) % buckets)
    pad = [0] * pad_size
    prev.extend(pad)
    splits = np.split(np.array(prev), bucket_size)
    x= np.max(splits, axis=1)
    y = max(graph_hist[sample:min(sample*2,len(graph_hist))])
    return x,y

# def convert_training_sample(graph_hist,sample,buckets):
#     bucket_size = graph_hist[sample]/buckets 
#     print bucket_size
#     if sample<bucket_size: return None,None
#     bounds = [(i*bucket_size,i*bucket_size+bucket_size) for i in range(buckets)]
#     print bounds
#     y = max(graph_hist[sample:min(sample*2,len(graph_hist))])
#     x = np.array(map(lambda i: max(graph_hist[i[0]:i[1]]),bounds))
#     return x,y

def convert_training_graph_history(graph_hist,buckets,num_samples=100):
    x=[]
    y=[]
    indicies = np.random.choice(graph_hist,num_samples,replace=False)
    for sample in indicies:
        x_1,y_1 = convert_training_sample(graph_hist,sample,buckets)
        if not x_1 is None:
            x.append(x_1)
            y.append(y_1)
    return x,y
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

    # # er=nx.erdos_renyi_graph(100,0.15)
    # # ws=nx.watts_strogatz_graph(10000,3,0.1)
    # # tree = nx.balanced_tree(2,10)
    # complete = nx.complete_graph(100)
    # # ba=nx.barabasi_albert_graph(100,5)
    # # red=nx.random_lobster(100,0.9,0.9)    
    # nx.draw(complete)
    # plt.show()
    # graph = complete
    # print "----------------------------testing regular stack----------------------------"
    # regular_stack = DynamicArray()
    # cp = cProfile.Profile()
    # cp.enable()
    # answer = dfs_iterative(graph, 1,regular_stack)
    # cp.disable()
    # cp.print_stats()
    print "----------------------------testing learned stack----------------------------"
    train = []
    num_epochs = 5
    for i in range(num_epochs):
        graph = nx.random_lobster(100,0.9,0.9)  
        learned_stack = LearnedDynamicArray()
        answer = dfs_iterative(graph, 1,learned_stack)
        train.append(learned_stack.history_n)
        # plt.figure()
        # plt.plot(learned_stack.history_n)
        # plt.plot(learned_stack.history_capacity)
        # plt.show()

    train_x,train_y = create_dataset(train)
    print train_x,train_y
    # print np.shape(train_x),np.shape(train_y)
    # model = build_lstm_model(np.expand_dims(train_x,2),np.expand_dims(train_y,2), batch_size=2, num_epochs=10,verbose=0)
    # predict= model.predict(pad_sequences([train[-1]],maxlen=padding_len))
    # print predict

    # complete = nx.complete_graph(100)
    # cp = cProfile.Profile()
    # cp.enable()
    # answer = dfs_iterative(graph, 1,learned_stack)
    # cp.disable()
    # cp.print_stats()
    # plt.figure()
    # plt.plot(learned_stack.history_n)
    # plt.plot(learned_stack.history_capacity)
    # plt.show()