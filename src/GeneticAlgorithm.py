import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_predict, ShuffleSplit
from sklearn.svm import SVC
from xgboost import XGBClassifier
from parallel_cv import train_test


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

def fitness_score(population, model, kfold, X, y, n_jobs):
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
	precision_list, recall_list,f1_list,accuracy_list,auc_list,confusion_matrices,mcc_list = [],[],[],[],[],[],[]
	for chromosome in population:
		scores = train_test(model, X.iloc[:,chromosome], y, kfold, n_jobs)
		accuracy_list.append(scores['Accuracy'])
		auc_list.append(scores['AUC'])
		precision_list.append(scores['Precision'])
		recall_list.append(scores['Recall'])
		f1_list.append(scores['F1-score'])
		mcc_list.append(scores['MCC'])

	fitness = np.array(accuracy_list) + np.array(f1_list)+ np.array(auc_list)
	penalized = fitness ** 2
	weights = penalized / np.sum(penalized)
	sorted_indices = np.argsort(fitness)[::-1]

	return (
	list(np.array(accuracy_list)[sorted_indices]),
	list(np.array(population)[sorted_indices, :]),
	list(np.array(precision_list)[sorted_indices]),
	list(np.array(recall_list)[sorted_indices]),
	list(np.array(f1_list)[sorted_indices]),	
	list(np.array(auc_list)[sorted_indices]),
	list(np.array(mcc_list)[sorted_indices]),
	fitness[sorted_indices],
	weights[sorted_indices])

# ------------------------------------------------------------
#                         Routllet Selection
# ------------------------------------------------------------
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

# ------------------------------------------------------------
#                         Crossover
# ------------------------------------------------------------
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

# ------------------------------------------------------------
#                         Mutation
# ------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------
#                            Generations / GA main loop
# ----------------------------------------------------------------------------------------
def generations(size, n_feat, crossover_rate, mutation_rate, max_gen, model, kfold, X, y, n_jobs):
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
	# History
	best_chromo, best_acc, best_precision, best_recall, best_f1, best_auc, best_mcc, len_best_chromo = [],[],[],[],[],[],[],[]

	# Initialize population
	population_nextgen = initialization_of_population(size, n_feat)

	# Initialize persistent elite outside generations loop
	persistent_elite = []  # list of tuples: (chromosome, fitness)
	elitism_n = 2

	#GA main loops
	for gen in range(max_gen):
		accuracy, pop_sorted, precision, recall, f1, auc, mcc, fitness, weights  = fitness_score(population_nextgen, model, kfold, X, y, n_jobs)
		
		# Update persistent elite
		for i in range(elitism_n):
			elite_record = {"chromo": pop_sorted[i],
			"fitness": fitness[i],
			"accuracy": accuracy[i],
			"precision": precision[i],
			"recall": recall[i],
			"f1": f1[i],
			"auc": auc[i],
			"mcc": mcc[i]}
			# If elite list not full → just add
			if len(persistent_elite) < elitism_n:
				persistent_elite.append(elite_record)
			else:
				# Replace worst elite if this chromosome is stronger
				worst_idx = np.argmin([e["fitness"] for e in persistent_elite])
				if elite_record["fitness"] > persistent_elite[worst_idx]["fitness"]:
					persistent_elite[worst_idx] = elite_record
		elite_chromosomes = [e["chromo"] for e in persistent_elite]

		# sort elites by fitness descending	
		sorted_elites = sorted(persistent_elite, key=lambda x: x["fitness"], reverse=True)
		best_elite = sorted_elites[0]
		len_c = len(np.where(best_elite["chromo"])[0])
	
		# Record best chromosome info		
		best_chromo.append(best_elite["chromo"])
		best_acc.append(best_elite["accuracy"])
		best_precision.append(best_elite["precision"])
		best_recall.append(best_elite["recall"])
		best_f1.append(best_elite["f1"])
		best_auc.append(best_elite["auc"])
		best_mcc.append(best_elite["mcc"])
		len_best_chromo.append(len(np.where(best_elite["chromo"] == 1)[0]))		

		print('Generation {}: Best Score - AUC: {} - Accuracy: {} - F1-score: {} - MCC: {} - Length: {}'.format(gen,best_elite["auc"],\
                best_elite["accuracy"],best_elite["f1"], best_elite["mcc"],len_c))

		# ------------------------------------------------------------
		#                           Selection
		# ------------------------------------------------------------
		elitism_n = 2   # keep best 2
		k_select = size - elitism_n
		pop_after_sel = selection(pop_sorted, weights, size)

		# ------------------------------------------------------------
		#                     Crossover + Mutation
		# ------------------------------------------------------------

		children = []
		num_pairs = k_select
		for i in range(0, num_pairs, 2):
			p1, p2 = pop_after_sel[i], pop_after_sel[i+1]
			for c in crossover(p1, p2, crossover_rate):
				mutation(c, mutation_rate)
				children.append(c)
	
		# ------------------------------------------------------------
		#                    Form next generation
		# ------------------------------------------------------------
		population_nextgen = elite_chromosomes + children

	return best_chromo, best_acc,best_precision, best_recall, best_f1, best_auc, best_mcc, len_best_chromo

