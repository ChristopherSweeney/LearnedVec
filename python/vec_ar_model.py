from standard_vector import DynamicArray
from learned_standard_vector import LearnedDynamicArray
import numpy as np
from igraph import *
import cProfile
import matplotlib.pyplot as plt
import networkx as nx
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from statsmodels.tsa.arima_model import ARIMA
from pandas import Series
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

def create_dataset(data,prediction_horizon=20,padding_len=200):
    data_y = np.zeros((len(data),padding_len))
    data_x = np.zeros((len(data),padding_len))
    for i,sample in enumerate(data):
        x,y = convert_training_sample(sample,prediction_horizon,padding_len)
        data_y[i,:]=x
        data_x[i,:]=y
    return data_x, data_y

def convert_training_sample(sample,prediction_horizon,padding_len):
    x=pad_sequences([sample[:-prediction_horizon]],maxlen=padding_len)
    y=pad_sequences([sample[prediction_horizon:]],maxlen=padding_len)
    return x,y

def train_ARIMA_model(P, D, Q):
	model = ARIMA(Actual, order=(P, D, Q))
	model_fit = model.fit(disp=0)
	prediction = model_fit.forecast()[0]
	return prediction

def build_lstm_model(train_x,train_y, batch_size=4, num_epochs=10,verbose=0):
    model = Sequential()
    model.add(LSTM(64, activation='relu',stateful=False,return_sequences=True,input_shape=(200,1)))
    # model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mse', optimizer='adam')
    print model.summary()
    model.fit(train_x,train_y, batch_size=batch_size, epochs=num_epochs,verbose=0)
    return model

# def generate_gaussian_queries_append(num):

# def generate_linear_queries_append(start_size,num_queries):
# 		return append()start_size,start_size+num_queries)

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
    num_epochs = 2
    padding_len = 200
    train = []
    for i in range(num_epochs):
        graph = nx.complete_graph(100)
        learned_stack = LearnedDynamicArray()
        answer = dfs_iterative(graph, 1,learned_stack)
        train.append(learned_stack.history_n)
    # train_x,train_y = create_dataset(train,prediction_horizon=20)
    model = AR(train[0])
    model_fit = model.fit()
    print('Lag: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)
    predictions = model_fit.predict(start=45, end=len(train[0]), dynamic=True)
    print predictions
    # error = mean_squared_error(train[45:100], predictions)
    # print('Test MSE: %.3f' % error)
    # plot results
    plt.plot(train[0])
    plt.plot(predictions, color='red')
    plt.show()
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