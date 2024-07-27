import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_predict, ShuffleSplit
from sklearn.svm import SVC
from xgboost import XGBClassifier
from Training import train_test

def initialization_of_population(size, n_feat):
	"""
	Initialize the population with random chromosomes.
	Parameters:
		size (int): Size of the population.
		n_feat (int): Number of features.
	Returns:
		population (list): Initialized population.
	"""
	population = []
	for _ in range(size):
		chromosome = np.ones(n_feat, dtype=bool)  # Use `dtype=bool` instead of `dtype=np.bool`
		chromosome[:int(0.3 * n_feat)] = False
		np.random.shuffle(chromosome)
		population.append(chromosome)
	return population

def fitness_score(population, model, kfold, X, y):
	"""
	Evaluate the fitness score of the population.
	Parameters:
		population (list): Population of chromosomes.
		model (object): Machine learning model.
		kfold (int): Number of folds for cross-validation.
		X (array-like): Feature set.
		y (array-like): Target labels.
	Returns:
		tuple: Evaluation scores and sorted population.
	"""
	cv = StratifiedKFold(n_splits=kfold, shuffle=True)
	precision_list, recall_list,f1_list,accuracy_list,auc_list,confusion_matrices,mcc_list = [],[],[],[],[],[],[]
	for chromosome in population:
		scores = train_test(model, X.iloc[:,chromosome], y, kfold)
		accuracy_list.append(scores['Accuracy'])
		auc_list.append(scores['AUC'])
		precision_list.append(scores['Precision'])
		recall_list.append(scores['Recall'])
		f1_list.append(scores['F1-score'])
		mcc_list.append(scores['MCC'])

	combined_score = np.array(accuracy_list) + np.array(f1_list)
	weights = combined_score / np.sum(combined_score)
	sorted_indices = np.argsort(combined_score)[::-1]
	
	return (
	list(np.array(accuracy_list[sorted_indices])),
	list(np.array(population)[sorted_indices, :]),
	list(np.array(precision_list)[sorted_indices]),
	list(np.array(recall_list)[sorted_indices]),
	list(np.array(f1_list)[sorted_indices]),	
	list(np.array(auc_list)[sorted_indices]),
	list(np.array(mcc_list)[sorted_indices]),
	list(weights[sorted_indices]))
	
def selection(pop_after_fit, weights, k):
	"""
	Select chromosomes based on fitness weights.
	Parameters:
	 	pop_after_fit (list): Population after fitness evaluation.
		weights (array-like): Fitness weights.
		k (int): Number of chromosomes to select.
	Returns:
		list: Selected population.
	"""
	selected_indices = np.random.choice(len(pop_after_fit), size=k, replace=True, p=weights)
	return [pop_after_fit[i] for i in selected_indices]

def crossover(p1, p2, crossover_rate):
	"""
	Perform crossover between two parent chromosomes.
	Parameters:
		p1 (array-like): First parent chromosome.
		p2 (array-like): Second parent chromosome.
		crossover_rate (float): Crossover rate.
	Returns:
		list: Two offspring chromosomes.
	"""
	c1, c2 = p1.copy(), p2.copy()
	if random.random() < crossover_rate:
		pt = random.randint(1, len(p1)-2)
		c1 = np.concatenate((p1[:pt], p2[pt:]))
		c2 = np.concatenate((p2[:pt], p1[pt:]))
	return [c1, c2]

def mutation(chromosome, mutation_rate):
	"""
	Mutate a chromosome based on mutation rate.
	Parameters:
		chromosome (array-like): Chromosome to mutate.
		mutation_rate (float): Mutation rate.
	Returns:
		None
	"""
	for i in range(len(chromosome)):
		if random.random() < mutation_rate:
			chromosome[i] = not chromosome[i]

def generations(size, n_feat, crossover_rate, mutation_rate, max_gen, model, kfold, X, y):
	"""
	Run the genetic algorithm for a specified number of generations.
	Parameters:
		size (int): Population size.
		n_feat (int): Number of features.
		crossover_rate (float): Crossover rate.
		mutation_rate (float): Mutation rate.
		max_gen (int): Maximum number of generations.
		model (object): Machine learning model.
		kfold (int): Number of folds for cross-validation.
		X (array-like): Feature set.
		y (array-like): Target labels.
	Returns:
		tuple: Best chromosomes and their evaluation scores over generations.
	"""
	best_chromo, best_acc, best_precision, best_recall, best_f1, best_auc, best_mcc, len_best_chromo = [],[],[],[],[],[],[],[]
	population_nextgen = initialization_of_population(size, n_feat)
	for gen in range(max_gen):
		accuracy, pop_after_fit, precision, recall, f1, auc, mcc, weights  = fitness_score(population_nextgen, model, kfold, X, y)
		acc, precision, recall, f1, auc, mcc = accuracy[0], precision[0], recall[0], f1[0], auc[0], mcc[0]
		len_c = len(np.where(pop_after_fit[0])[0])
		print('Generation {}: Best Score - acc: {} - F1-score: {} - mcc: {} - len: {}'.format(gen, acc,\
		f1,mcc,len_c))
		k = size - 2	
		pop_after_sel = selection(pop_after_fit, weights, k)

		children = []
		for i in range(0, len(pop_after_sel), 2):
			p1, p2 = pop_after_sel[i], pop_after_sel[i+1]
			for c in crossover(p1, p2, crossover_rate):
				mutation(c, mutation_rate)
				children.append(c)
		pop_after_mutated = children
		population_nextgen = pop_after_fit[:2] + pop_after_mutated

		best_chromo.append(pop_after_fit[0])
		best_acc.append(acc)
		best_precision.append(precision)
		best_recall.append(recall)
		best_f1.append(f1)
		best_auc.append(auc)
		best_mcc.append(mcc)
		len_best_chromo.append(len_c)
	    
	return best_chromo, best_acc,best_precision, best_recall, best_f1, best_auc, best_mcc, len_best_chromo

