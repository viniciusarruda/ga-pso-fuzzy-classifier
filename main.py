import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn import datasets
from genetic_algorithm import genetic_algorithm

# https://github.com/nathanrooy/particle-swarm-optimization
import pso_simple

def normalize_dataset(dataset):
	# Normalize the dataset to [0, 1]
	min_arr = np.amin(dataset, axis=0)
	return (dataset - min_arr) / (np.amax(dataset, axis=0) - min_arr)


def evaluate_new_fuzzy_system(w1, w2, w3, w4, data, target):

	input_universe = np.linspace(0, 1, 100)

	x = []

	for w in [w1, w2, w3, w4]:
		
		d = {'s': fuzz.trimf(input_universe, [0.0, 0.0, w]),
		     'm': fuzz.trimf(input_universe, [0.0, w, 1.0]),
			 'l': fuzz.trimf(input_universe, [w, 1.0, 1.0])}

		x.append(d)

	h, w = data.shape

	hits = 0.0

	for p in range(0, h):
		_x1 = data[p, 0]
		_x2 = data[p, 1]
		_x3 = data[p, 2]
		_x4 = data[p, 3]

		x1_short = fuzz.interp_membership(input_universe, x[0]['s'], _x1)
		x1_middle = fuzz.interp_membership(input_universe, x[0]['m'], _x1)
		x1_long = fuzz.interp_membership(input_universe, x[0]['l'], _x1)

		x2_short = fuzz.interp_membership(input_universe, x[1]['s'], _x2)
		x2_middle = fuzz.interp_membership(input_universe, x[1]['m'], _x2)
		x2_long = fuzz.interp_membership(input_universe, x[1]['l'], _x2)

		x3_short = fuzz.interp_membership(input_universe, x[2]['s'], _x3)
		x3_middle = fuzz.interp_membership(input_universe, x[2]['m'], _x3)
		x3_long = fuzz.interp_membership(input_universe, x[2]['l'], _x3)

		x4_short = fuzz.interp_membership(input_universe, x[3]['s'], _x4)
		x4_middle = fuzz.interp_membership(input_universe, x[3]['m'], _x4)
		x4_long = fuzz.interp_membership(input_universe, x[3]['l'], _x4)

		is_setosa = np.fmin(np.fmax(x3_short, x3_middle), x4_short)
		is_versicolor = np.fmax(np.fmin(np.fmin(np.fmin(np.fmax(x1_short, x1_long), np.fmax(x2_middle, x2_long)), np.fmax(x3_middle, x3_long)),x4_middle), np.fmin(x1_middle, np.fmin(np.fmin(np.fmax(x2_short, x2_middle),x3_short), x4_long)))
		is_virginica = np.fmin(np.fmin(np.fmax(x2_short, x2_middle), x3_long), x4_long)

		result = np.argmax([is_setosa, is_versicolor, is_virginica])

		if result == target[p]:
			hits += 1.0

	return hits / h




def create_new_fuzzy_system(w1, w2, w3, w4):
	
	# universe = np.arange(0.0, 1.0, 0.01) # this will not include the 1.0 ...
	universe = np.linspace(0, 1, 100)

	x1 = ctrl.Antecedent(universe, 'x1')
	x2 = ctrl.Antecedent(universe, 'x2')
	x3 = ctrl.Antecedent(universe, 'x3')
	x4 = ctrl.Antecedent(universe, 'x4')

	### creating the membership functions
	for x, w in zip([x1, x2, x3, x4], [w1, w2, w3, w4]):

		x['s'] = fuzz.trimf(universe, [0.0, 0.0, w])
		x['m'] = fuzz.trimf(universe, [0.0, w, 1.0])
		x['l'] = fuzz.trimf(universe, [w, 1.0, 1.0])

	output = ctrl.Consequent([0, 1, 2], 'classification')
	output.automf(names=['setosa', 'versicolor', 'virginica'])

	# output.view()
	# plt.show(block=True)

	# the rules
	setosa_rule = ctrl.Rule(antecedent=((x3['s'] | x3['m']) & x4['s']),
		                    consequent=output['setosa'],
		                    label='Setosa Rule')

	versicolor_rule = ctrl.Rule(antecedent=(((x1['s']  | x1['l']) & (x2['m'] | x2['l']) & (x3['m'] | x3['l']) & (x4['m']) ) | (x1['m'] & (x2['s'] | x2['m']) & x3['s'] & x4['l'])),
	    						consequent=output['versicolor'],
	    						label='Versicolor Rule')
	
	virginica_rule = ctrl.Rule(antecedent=((x2['s'] | x2['m']) & x3['l'] & x4['l']),
							   consequent=output['virginica'],
							   label='Virginica Rule')
	
	# our main control system
	system_ctrl = ctrl.ControlSystem([setosa_rule, versicolor_rule, virginica_rule])

	# we need a simulator
	classifier = ctrl.ControlSystemSimulation(system_ctrl)

	return classifier


def custom_deffuzz(v):
	# Retrieves the class with less error
	return np.argmin(np.abs([v, v - 1.0, v - 2.0]).T, axis=1)
	# return np.argmin(np.abs([v - 0.5, v - 1.0, v - 1.5]).T, axis=1)


def evaluate_fuzzy_classifier(classifier, samples, labels):

	classifier.input['x1'] = samples[:, 0]
	classifier.input['x2'] = samples[:, 1]
	classifier.input['x3'] = samples[:, 2]
	classifier.input['x4'] = samples[:, 3]

	classifier.compute()

	c = classifier.output['classification']

	return (custom_deffuzz(c) == labels).mean()


data, target = None, None
def create_and_evaluate(w):

	classifier = create_new_fuzzy_system(w[0], w[1], w[2], w[3])
	return 1.0 - evaluate_fuzzy_classifier(classifier, data, target)


def main():
	
	iris = datasets.load_iris()
	normalized_iris = normalize_dataset(iris.data)
	n_features = normalized_iris.shape[1]

	global data, target
	data = normalized_iris
	target = iris.target

	# Test Fuzzy
	w1, w2, w3, w4 = 0.07, 0.34, 0.48, 0.26
	# w1, w2, w3, w4 = 0.17391253, 0.08085587, 0.93455427, 0.6963448
	# w1, w2, w3, w4 = 0.20204260148620853, 1, 0.7445948687715257, 0.6117803008912537
	classifier = create_new_fuzzy_system(w1, w2, w3, w4)
	print('Old:')
	print(evaluate_fuzzy_classifier(classifier, normalized_iris, iris.target))
	print('New:')
	print(evaluate_new_fuzzy_system(w1, w2, w3, w4, data, target))

	# GA
	# best, fbest = genetic_algorithm(fitness_func=create_and_evaluate, dim=n_features, n_individuals=10, epochs=30)
	# print(best, fbest)

	# PSO
	# initial=[0.5, 0.5, 0.5, 0.5]             
	# bounds=[(0, 1), (0, 1), (0, 1), (0, 1)] 
	# pso_simple.minimize(create_and_evaluate, initial, bounds, num_particles=10, maxiter=30, verbose=True)


if __name__ == '__main__':
	main()