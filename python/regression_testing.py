from standard_vector import DynamicArray
from learned_standard_vector import LearnedDynamicArray
import numpy as np
from igraph import *
import cProfile
import matplotlib.pyplot as plt
import networkx as nx
from scipy.signal import find_peaks
from sklearn.svm import SVR

#samples per graph greater than graph duration???
def create_dataset(data,buckets=50,samples_per_graph =100): #data [history_n], history_n :[1,2,3]
    data_y = np.zeros(len(data)*samples_per_graph)
    data_x = np.zeros((len(data)*samples_per_graph,buckets))
    for i,graph_hist in enumerate(data):
        print "ingesting graph "+str(i)
        x,y,choices = convert_training_graph_history(graph_hist,buckets,samples_per_graph)
        data_y[i*samples_per_graph:i*samples_per_graph+samples_per_graph]= y
        data_x[i*samples_per_graph:i*samples_per_graph+samples_per_graph,:]= x
    return data_x.astype(np.float64), data_y.astype(np.float64),choices

# per graph
def convert_training_graph_history(graph_hist,buckets,num_samples=100):
    y = np.zeros(num_samples)
    x = np.zeros((num_samples,buckets))
    indicies = np.random.choice(np.arange(len(graph_hist))[1:],num_samples,replace=False) #must have more data than samples
    for i,sample in enumerate(indicies):
        x_1,y_1 = convert_training_sample(graph_hist,sample,buckets)
        x[i,:] = x_1
        y[i] = y_1
    return x,y,indicies

# per sample
def convert_training_sample(graph_hist,sample,buckets):
    prev = graph_hist[:sample]
    x,y = None,None
    if sample>=buckets:
        bucket_size = int(sample / buckets)
        x = max_buckets(prev[sample%buckets:], buckets)
    else:
        x = np.zeros((buckets))
        x[buckets - sample:] = np.array(prev)
    y = get_y(graph_hist,sample)
    return x,y

def max_buckets(array, buckets):
    splits = np.split(np.array(array), buckets)
    x = np.max(splits, axis=1)
    return x

def get_y(graph_hist,sample):
    return max(graph_hist[sample:min(sample*2,len(graph_hist)-1)])

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
    num_epochs = 1
    for i in range(num_epochs):
        graph = nx.random_lobster(100,0.9,0.9)
        nx.draw(graph)
        plt.show()
        learned_stack = LearnedDynamicArray()
        answer = dfs_iterative(graph, 1,learned_stack)
        train.append(learned_stack.history_n)
        #regress on mulitplicitve factor
        #decay factor
        train_x,train_y,choices = create_dataset(train)
        print train_x[0], train_y[0]
        clf = SVR( C=1.0, epsilon=0.2)
        clf.fit(train_x, train_y)
        predictions = clf.predict([train_x[0]])

        plt.figure()
        plt.plot(learned_stack.history_n)
        plt.plot(learned_stack.history_capacity)
        plt.scatter(np.arange(len(train_x[0]))*choices[0]/50,train_x[0],color="g")
        plt.scatter(choices[0], train_y[0],color="r")
        plt.scatter(choices[0], predictions,color="b")
        plt.show()

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