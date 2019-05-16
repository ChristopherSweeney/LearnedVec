import numpy as np
import networkx as nx
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences

def build_lstm_model(lstm_units, buckets=10): #lstm stateful sees every element
    model = Sequential()
    model.add(LSTM(lstm_units, activation='relu',stateful=False,input_shape=(buckets,1)))
    # model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer='rmsprop',lr=0.01)
    print model.summary()
    return model

#samples per graph greater than graph duration???
def create_dataset(data, buckets=20, samples_per_graph =100, look_ahead_rate = 2): #data [history_n], history_n :[1,2,3]
    data_y = np.zeros(len(data)*samples_per_graph)
    data_x = np.zeros((len(data)*samples_per_graph,buckets))
    choices = []
    for i,graph_hist in enumerate(data):
        print "ingesting graph "+str(i)
        x,y,choice = convert_training_graph_history(graph_hist,buckets,look_ahead_rate,samples_per_graph)
        data_y[i*samples_per_graph:i*samples_per_graph+samples_per_graph]= y
        data_x[i*samples_per_graph:i*samples_per_graph+samples_per_graph,:]= x
        choices.append(choice)
    return data_x.astype(np.float64), data_y.astype(np.float64),choices

# per graph
def convert_training_graph_history(graph_hist,buckets,look_ahead_rate,num_samples):
    y = np.zeros(num_samples)
    x = np.zeros((num_samples,buckets))
    indicies = np.random.choice(np.arange(len(graph_hist))[1:],num_samples,replace=False) #must have more data than samples
    for i,sample in enumerate(indicies):
        x_1 = convert_training_sample_x(graph_hist,sample,buckets)
        y_1 = convert_training_sample_y(graph_hist,sample,look_ahead_rate)
        x[i,:] = x_1
        y[i] = y_1
    return x,y,indicies

# per sample
def convert_training_sample_x(graph_hist,sample,buckets):
    prev = graph_hist[:sample]
    x = None
    n = float(graph_hist[sample])
    #edge case where index is 0 and n is 0
    #there are enough samples to populate buckets. TODO use DP to speed up max_buckets   
    if sample>=buckets:
        bucket_size = int(sample / buckets)
        #if not even split take most recent full buckets
        x = max_buckets(prev[sample%buckets:], buckets)
        x = x / (n+1)
    #if we are at the beggining of the array    
    else:
        x = np.zeros((buckets))
        x[buckets - sample:] = np.array(prev)/(n+1) #is this the right thing to do?
    return x

def convert_training_sample_y(graph_hist,sample,look_ahead_rate):
    n = float(graph_hist[sample])
    y = get_y(graph_hist,sample,look_ahead_rate)
    y = y / (n+1)
    return y

def max_buckets(array, buckets):
    splits = np.split(np.array(array), buckets)
    x = np.max(splits, axis=1)
    return x

def get_y(graph_hist,sample,look_ahead_rate):
    n = graph_hist[sample]
    look_ahead = sample + int(np.ceil((n+1) * (look_ahead_rate - 1)))
    look_ahead = len(graph_hist) if (len(graph_hist)-1) <= look_ahead else look_ahead
    return max(graph_hist[sample:look_ahead]) #be aware of edge case where the max could happen at sample
    #for the last half of time this doesnt work as 
    #you are always prediciting 0 thinkin about that, we are also overfitting to stacks that run out of elements???
    #also maybe think about not max but average

#source: 
#https://www.koderdojo.com/blog/depth-first-search-in-python-recursive-and-non-recursive-programming
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