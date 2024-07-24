import os
import pandas as pd
from sklearn.svm import SVC
from Genetic_Algorithm import generations

# Configuration
kfold = 10
model = SVC(kernel='linear', probability=True)
antibiotics = ["tobramycin", "ceftazidime"]
input_folder = '/nas/users/duyen/pseudomonas/SNP_Clstr/data/SNP/'
output_folder = f'{input_folder}/results1'

# Ensure results directory exists
os.makedirs(output_folder, exist_ok=True)

# Running GA for each antibiotic
for antibiotic in antibiotics:
	try:
		print(f"Processing {antibiotic}")
		file_path = f"{input_folder}trainGA_{antibiotic.lower()}"
		df = pd.read_csv(f'{file_path}.csv')
		X = df.iloc[:,:-1]
		y = df.iloc[:,-1]

		best_chromo, best_acc, best_precision, best_recall, best_f1, best_auc, best_mcc, len_best_chromo = generations(size=50, n_feat=X.shape[1], crossover_rate=0.8, mutation_rate=0.05, max_gen=5, model = model, kfold = kfold, X=X, y=y)

		generation_results = {
		'Accuracy': best_acc,
		'Precision': best_precision,
		'Recall': best_recall,
		'F1 Score': best_f1,
		'AUC': best_auc,
		'MCC': best_mcc,
		'Length': len_best_chromo}

		df_generation = pd.DataFrame(generation_results)
		df_generation.to_csv(f'{output_folder}/{antibiotic}_genresults.csv',index=False)
		print(df_generation)
	except Exception as e:
		print(f"An error occurred while processing {antibiotic}: {e}")
