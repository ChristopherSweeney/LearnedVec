import sys
import ctypes 
import numpy as np

class LearnedDynamicArray(object): 
    ''' 
    DYNAMIC ARRAY CLASS (Similar to Python List) 
    '''
      
    def __init__(self,capacity=1,model=None,buckets = 10): 
        self.n = 0 # Count actual elements (Default is 0) 
        self.capacity = capacity # Default Capacity 
        self.A = self.make_array(self.capacity) 
        self.history_n=[]
        self.history_capacity=[]
        self.model = model
        self.buckets = buckets
        self.prediction_points=[]
        self.prediction_vals=[]

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
            if not self.model:
                self._resize(int(1/2. * self.capacity))
            else:
                horizon =self.predict_horizon()
                if self.n >= int(horizon)+2:
                    self._resize(1/2 * self.capacity)
                    print "error"
                else:
                    self._resize(int(horizon)+2)#buffer
        return ele

    def predict_horizon(self):
        history = self.history_n
        size = len(history)
        if size>=self.buckets:
            bucket_size = int(size / self.buckets)
            x = self.max_buckets(history[size%self.buckets:])
        else:
            x = np.zeros((self.buckets))
            x[self.buckets - size:] = np.array(self.history_n)
        return self.model.predict([x])

    def max_buckets(self,array):
        splits = np.split(np.array(array), self.buckets)
        x = np.max(splits, axis=1)
        return x

    def append(self, ele): 
        """ 
        Add element to end of the array 
        """
        if self.n == self.capacity: 
            if not self.model:
                self._resize(2 * self.capacity)
            else:
                horizon = self.predict_horizon()
                print "horizin ", horizon
                print "capacity", self.capacity
                print "n ", self.n

                if self.n >= int(horizon)+2:
                    self._resize(2 * self.capacity)
                    print "error"
                else:
                    self._resize(int(horizon)+2)#buffer

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



