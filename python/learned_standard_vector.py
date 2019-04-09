import sys
import ctypes 
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM

class LearnedDynamicArray(object): 
    ''' 
    DYNAMIC ARRAY CLASS (Similar to Python List) 
    '''
      
    def __init__(self,capacity=1,LSTM_model=None): 
        self.n = 0 # Count actual elements (Default is 0) 
        self.capacity = capacity # Default Capacity 
        self.A = self.make_array(self.capacity) 
        self.history_n=[]
        self.history_capacity=[]
        self.LSTM_model = LSTM_model
          
    def __len__(self): 
        """ 
        Return number of elements sorted in array 
        """
        return self.n 
      
    def __getitem__(self, k): 
        """ 
        Return element at index k 
        """
        if not 0 <= k <self.n: 
            # Check it k index is in bounds of array 
            return IndexError('K is out of bounds !')  
          
        return self.A[k] # Retrieve from the array at index k 
    
    def update_history(self):
        self.history_n.append(self.n)
        self.history_capacity.append(self.capacity)

    def pop(self):
        """ 
        Remove element from the end of the array 
        """
        if self.n<1:
            return None
        ele = self.A[self.n-1]
        self.n-=1
        self.update_history()
        if 4*self.n < self.capacity and self.capacity>1: 
            # halve capacity
            self._resize(int(1/2. * self.capacity))
        return ele
    
    def append(self, ele): 
        """ 
        Add element to end of the array 
        """
        if self.n == self.capacity: 
            # Double capacity if not enough room 
            self._resize(2 * self.capacity)  
        self.A[self.n] = ele # Set self.n index to element 
        self.n += 1
        self.update_history()

          
    def _resize(self, new_cap): 
        """ 
        Resize internal array to capacity new_cap 
        """
          
        B = self.make_array(new_cap) # New bigger array 
        for k in range(self.n): # Reference all existing values 
            B[k] = self.A[k] 
              
        self.A = B # Call A the new bigger array 
        self.capacity = new_cap # Reset the capacity 
          
    def make_array(self, new_cap): 
        """ 
        Returns a new array with new_cap capacity 
        """
        return (new_cap * ctypes.py_object)() 

if __name__ == '__main__':
    arr = LearnedDynamicArray()

    print("\ntesting appending\n")
    for i in range(100):
        print(i)
        arr.append(i)
        print("capacity: ", arr.capacity)
        print("size: ", len(arr))

    print("\ntesting popping\n")
    for i in range(100):
        e = arr.pop()
        print(e)
        print("capacity: ", arr.capacity)
        print("size: ", len(arr))



