"""
Program: ants_bin.py
Programed By: Ryan Bagby, Lennard Mussau, Michael Knapp
Discription: Neural Network with back propagation using ants TSA solution to randomize our hidden layer.
Trace Folder: knapp296,
"""

#--------------------------------------Imports-------------------------------------------
import numpy as np
import pandas as pd
from math import sqrt
from random import randint, random
import itertools as it

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale 

#any more imports go here_____

#---------------------------------------------------------------------------------------------
#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation

#-------------------------------------------Variables --------------------------------------------------------------

#global varibles go here_______


#----------------------------------------------------------------------------------------------------------------

#------------------------------------Classes/Functions-----------------------------------------------------------------
def accuracy(hidden_shape, X, y):
	classifier = MLPClassifier(hidden_layer_sizes=hidden_shape, solver='adam', activation='relu', 
					max_iter=50, batch_size=125)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, train_size=0.03)

	classifier = classifier.fit(X_train, y_train)
	score = classifier.score(X_test, y_test)

	#print('{} has accuracy of {}%'.format(hidden_shape, round(score,4)*100))

	return score


def initialize_pheromone(num_bits, init_phero):
    pheromones = [[[init_phero for j in range(2)] for k in range(2)] for l in range(num_bits)]
 
    return pheromones



def get_choices(X, y, last_index, start_layer, start_shape, next_shape, pheromone, heur, history, min_max, num_bits):
    choices = []

    for bit in range(2):
   		shape = [start_shape, next_shape]
   		prob = {'shape': shape}
   		prob['history']   = pheromone[last_index][start_layer[last_index]][bit] ** history
   		prob['accuracy']  = accuracy(shape, X, y) 
   		prob['cost']      = (1.0 - prob['accuracy']) * 100 if prob['accuracy'] < 1.0 else 1.0
   		prob['heuristic'] = (1.0 / prob['cost']) ** heur
   		prob['prob']      = prob['history'] * prob['heuristic']
   		prob['binary']    = bit
   		choices.append(prob)

    return choices


def prob_select(choices):
	print('PROBABILISTIC CHOICE \n')

	probs = [c['prob'] for c in choices]
	s = sum(probs)
	if s == 0:
		c = choices[randint(0, 1)]
		return c['binary']

	v = random()
	for c in choices:
		v -= c['prob']/s
		if v < 0.0:
			return c['binary']

	return choices[-1]['binary']



def greedy_select(choices):
	print('GREEDY CHOICE \n')

	probs = [c['prob'] for c in choices]
	i = np.argmax(probs)

	return choices[i]['binary']



def calc_shape(bin_list):
	shape = sum([bin_list[i]  << i for i in range(len(bin_list))])

	return shape


def stepwise_construct(X, y, start_layer, pheromone, heur, greed, min_max, num_bits):
	next_shape = 1
	bin_list = [next_shape]
	start_shape = calc_shape(start_layer)

	for j in range(1,num_bits):
		i = num_bits - j 

		cand = get_choices(X, y, i, start_layer, start_shape, next_shape, pheromone, heur, 1.0, min_max, num_bits)

		greedy = random() <= greed
		next_bit = greedy_select(cand) if greedy else prob_select(cand)
		next_shape += next_bit << i + 1

		bin_list.append(next_bit)

	bin_list.reverse()
	candidate = {'binary': bin_list}
	#print('Start layer: {} & binary list: {}'.format(start_layer, bin_list))
	candidate['shape'] = [start_shape, next_shape]
	#print('Stepwise candidate shape is {} \n'.format(candidate['shape']))
	candidate['accuracy'] = accuracy(candidate['shape'], X, y)

	return candidate['shape'], candidate['binary'], candidate['accuracy']



def global_update_phero(start_layer, pheromone, best, decay, num_bits):
	print('Global pheromone update.')
	h1 = start_layer
	h2 = best['binary']
	for bit in range(num_bits):
		value = ((1.0 - decay)*pheromone[bit][h1[bit]][h2[bit]]) + (decay * best['accuracy'])
		pheromone[bit][h1[bit]][h2[bit]] = value



def local_update_phero(start_layer, pheromone, candidate, local_phero, init_phero, num_bits):
	print('Local pheromone update.')
	h1 = start_layer
	h2 = candidate['binary']
	for bit in range(num_bits):
		value = ((1.0 - local_phero)*pheromone[bit][h1[bit]][h2[bit]]) + (local_phero*init_phero)
		pheromone[bit][h1[bit]][h2[bit]] = value



def search(X, y, layer_shapes, max_iters, num_ants, decay, heur, local_phero, greed, min_max, num_bits):
	b_hidden1 = layer_shapes[randint(0, len(layer_shapes)-1)]
	b_hidden2 = layer_shapes[randint(0, len(layer_shapes)-1)]

	best = {'binary': b_hidden2}

	b_hidden1 = calc_shape(b_hidden1)
	b_hidden2 = calc_shape(b_hidden2)

	best['shape'] = [b_hidden1, b_hidden2]
	best['accuracy'] = accuracy(best['shape'], X, y)

	print('Starting shape is {} with accuracy {}% \n'.format(best['shape'],round(best['accuracy'],4)*100))

	init_phero = 1.0/len(layer_shapes) 
	pheromone = initialize_pheromone(len(layer_shapes), init_phero)
    
	for i in range(max_iters):
 		for a in range(num_ants):
			print('Iteration {} of {} \n'.format(i+1, max_iters))
			print('Ant {} of {} \n'.format(a+1, num_ants))

			start_layer = layer_shapes[randint(0, len(layer_shapes)-1)]
			start_shape = calc_shape(start_layer)

			print('Starting layer shape is {} \n'.format(start_shape))
			
			candidate = {}
			candidate['shape'], candidate['binary'], candidate['accuracy'] = stepwise_construct(X, y, start_layer, pheromone, 
        																							heur, greed, min_max, num_bits)

			print('Chose new candidate with shape {} and accuracy of {}% \n'.format(candidate['shape'], round(candidate['accuracy'],4)*100))
			print('Current best is shape {} with accuracy {}%'.format(best['shape'], round(best['accuracy'],4)*100))
			best = candidate if candidate['accuracy'] > best['accuracy'] else best
			print('Now the best is shape {} with accuracy {}% \n'.format(best['shape'], round(best['accuracy'],4)*100))
			local_update_phero(start_layer, pheromone, candidate, local_phero, init_phero, num_bits)
			print('======================================================= \n')

		global_update_phero(start_layer, pheromone, best, decay, num_bits)
		print('======================================================= \n')


	print('Finished Search. \n')
	return best
#---------------------------------------Program Main---------------------------------------------------------------------
	
	# program main goes here_____________
	
# --- Problem constants
# heuristic; significance of historical choices; typically between 2 and 5
HEUR = 2.5
# pheromone influence factor
LOCAL_PHERO = 0.1
# likelihood of choosing greedily (instead of probabilistically)
GREED = 0.9

min_max = (0, 300)

build_net = True

setup_range = range(min_max[0]+1, min_max[1])

multiplier = 1

num_bits = 6

layer_shapes = [list(a) for a in it.product([0, 1], repeat=num_bits)]
zero_vector = [0 for i in range(num_bits)]
layer_shapes.remove(zero_vector)

max_iters = 5

num_ants = 10

decay = 0.1

file = "EEG_Eye_Detection.csv"

data = pd.read_csv(file)

X = data[:].iloc[:, 0:14]
y = data[:].iloc[:, 14]

X = scale(X)

# --- Run ant colony search algorithm 
best = search(X, y, layer_shapes, max_iters, num_ants, decay, HEUR, LOCAL_PHERO, GREED, min_max, num_bits)

print('Shape of best network is {} \n'.format(best['shape']))

if build_net:
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=None)

	print('Starting to train network returned by ants. \n')
	classifier = MLPClassifier(hidden_layer_sizes=best['shape'], solver='adam', activation='relu', max_iter=200, batch_size=200)
	classifier = classifier.fit(X_train, y_train)

	scores = classifier.score(X_test, y_test)

	# Best MLP so far: hidden_layer_sizes=(262,18), solver='adam', activation='relu', max_iter=200, batch_size=200
	# Accuracy: 89.39% with test_size=0.1, train_size=None
	# 2nd Best: shape=(272,235) and accuracy=89.19%
	# 3rd Best: shape=(193,76) and accuracy=89.12%
	# 4th Best: shape=(160,94) and accuracy=88.92%
	print('After further training, network of shape {} has accuracy of {}%.'.format(best['shape'], round(scores,4)*100))

#---------------------------------------------------End of Progr_---------------------------------------------------------------------- 