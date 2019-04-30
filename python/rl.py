import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.svm import SVR
import util
import ctypes 
from learned_standard_vector import LearnedDynamicArray


class Environment(object):     
		def __init__(self, graph, start, capacity=1, buckets = 10): 
				self.n = 0 # Count actual elements (Default is 0) 
				self.capacity = capacity # Default Capacity. TODO maybe start with small constant default capacity????
				self.operations=0

				self.buckets = buckets
				self.A = self.make_array(self.capacity) 

				#bookeeping
				self.history_n=[]
				self.history_capacity=[]

				# setup DFS
				self.graph = graph
				self.start = start
				self.path = []
				self.append(self.start) #capacity starts at 1
				self.neighbors_to_add = []
				self.first_loop = True

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
		
		def _reward(self):
			"""
			Return reward
			"""
			pass

		def observation(self):
			"""
			Return state we want the agent to use to make a prediction
			"""
			while self.n>0 or self.first_loop:
				if self.n < int(.25 * self.capacity) and self.capacity>1: 
					return util.convert_training_sample_x(self.history_n,self.history_n[-1],self.buckets)
						#finished adding neighbors
				no_neighbors_left = len(self.neighbors_to_add)==0
				if no_neighbors_left: 
					vertex = self.pop()
					if vertex in self.path: #change to hash table???
						continue
					self.path.append(vertex)
				self.neighbors_to_add =self.neighbors_to_add if not no_neighbors_left else [i for i in nx.all_neighbors(graph,vertex)][::-1]
				num_neighbors = len(self.neighbors_to_add)
				for i in range(num_neighbors):
					if self.n == self.capacity: 
								#agent needs to resize up
						return util.convert_training_sample_x(self.history_n,self.history_n[-1],self.buckets)
					neighbor = self.neighbors_to_add.pop()
					self.append(neighbor)
					self.first_loop = False
				# TODO: include self.n as a feature
			return None

		def action(self, resize_rate):
			"""
			Take the action (does the resizing) of the array.
			Returns the reward associated with the taken action.
			"""
			self._resize(int(np.ceil(self.capacity * resize_rate)))
			# Continue the dfs until another action is needed
			# Return reward
			return self._reward()

		def reset(self):
			pass

		def update_history(self):
				self.history_n.append(self.n)
				self.history_capacity.append(self.capacity)

		def pop(self):
				""" 
				Remove element from the end of the array 
				"""
				self.operations+=1
				if self.n<1:
						return None
				ele = self.A[self.n-1]
				self.n-=1
				self.update_history()
			
				return ele

		def append(self, ele): 
				""" 
				Add element to end of the array 
				"""
				self.operations+=1
				self.A[self.n] = ele # Set self.n index to element 
				self.n += 1
				self.update_history()
					
		def _resize(self, new_cap): 
				""" 
				Resize internal array to capacity new_cap 
				"""
				B = self.make_array(new_cap) # New bigger array 
				self.operations+=self.n

				for k in range(self.n): # Reference all existing values 
						B[k] = self.A[k] 
							
				self.A = B # Call A the new bigger array 
				self.capacity = new_cap # Reset the capacity 
					
		def make_array(self, new_cap): 
				""" 
				Returns a new array with new_cap capacity 
				"""
				return (new_cap * ctypes.py_object)() 

		def get_state(self):
			print self.capacity,self.n,self.operations


if __name__ == '__main__':
	graph = nx.watts_strogatz_graph(1000,3,0.1)
	start = 1
	buckets = 10
	en = Environment(graph, start, buckets = buckets)
	for i in range(100):
		step = en.observation()
		print step

		if not step is None:
			if en.capacity==en.n:
				en.action(2)
			else:
				en.action(.5)
		else:
			print "finished graph"

	plt.figure()
	plt.plot(en.history_n)
	plt.plot(en.history_capacity)
	plt.show()



	print "----------------------------testing regular stack----------------------------"
	regular_stack = LearnedDynamicArray(default_resize_up = 2) #with no model, this is just a regular dynamic array
	answer = util.dfs_iterative(graph, 1,regular_stack)
	plt.figure()
	plt.plot(regular_stack.history_n)
	plt.plot(regular_stack.history_capacity)
	plt.show()
