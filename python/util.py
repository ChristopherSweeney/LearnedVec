import numpy as np
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences

def build_lstm_model(train_x,train_y, batch_size=4, num_epochs=10,verbose=0):
    model = Sequential()
    model.add(LSTM(64, activation='relu',stateful=False,return_sequences=True,input_shape=(200,1)))
    # model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer='adam')
    print model.summary()
    model.fit(train_x,train_y, batch_size=batch_size, epochs=num_epochs,verbose=0)
    return model

#samples per graph greater than graph duration???
def create_dataset(data,buckets=20,samples_per_graph =100): #data [history_n], history_n :[1,2,3]
    data_y = np.zeros(len(data)*samples_per_graph)
    data_x = np.zeros((len(data)*samples_per_graph,buckets))
    choices = []
    for i,graph_hist in enumerate(data):
        print "ingesting graph "+str(i)
        x,y,choice = convert_training_graph_history(graph_hist,buckets,samples_per_graph)
        data_y[i*samples_per_graph:i*samples_per_graph+samples_per_graph]= y
        data_x[i*samples_per_graph:i*samples_per_graph+samples_per_graph,:]= x
        choices.append(choice)
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