import argparse
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from GeneticAlgorithm import generations  # make sure this file is in src/
 
# ============================== Argument Parser ==============================
parser = argparse.ArgumentParser(description='Run GA for antibiotic.')
parser.add_argument('-antibiotic', type=str, required=True, help='Antibiotic name')
parser.add_argument('-gen_number', type=int, required=True, help='Number of generations')
parser.add_argument('-outdir', type=str, required=True, help='Output directory inside data/combine/')
args = parser.parse_args()

antibiotic = args.antibiotic
gen_number = args.gen_number
output_dir = args.outdir

# ============================== Paths ==============================
# Folder where data is located, relative to this script
ada_folder = os.path.join(os.path.dirname(__file__), "../data/combine/")
ada_folder = os.path.abspath(ada_folder) + "/"

# Create output folders
os.makedirs(os.path.join(ada_folder, output_dir), exist_ok=True)
os.makedirs(os.path.join(ada_folder, output_dir, "feature_set"), exist_ok=True)

# ============================== Config ==============================
kfold = 5
n_jobs = 30
model = SVC(kernel='linear', probability=True)

# ============================== Load Dataset ==============================
print(f"Running GA for antibiotic: {antibiotic}")
csv_path = os.path.join(ada_folder, f"ADA_combine_{antibiotic.lower()}.csv")
df = pd.read_csv(csv_path, dtype={'index': str}).set_index('index')
df.index.name = None
df.columns = df.columns.str.replace(r'^.*(?=Cluster)', '', regex=True)
df = df.astype(int)

X = df.drop('resistant_phenotype', axis=1)
y = df['resistant_phenotype']

# ============================== Run Genetic Algorithm ==============================
best_chromo, best_acc, best_precision, best_recall, best_f1, best_auc, best_mcc, len_best_chromo = generations(
    size=50,
    n_feat=X.shape[1],
    crossover_rate=0.8,
    mutation_rate=0.5,
    max_gen=gen_number,
    model=model,
    kfold=kfold,
    X=X,
    y=y,
    n_jobs=n_jobs
)

# ============================== Save Training Results ==============================
generation_results = {
    'Accuracy': best_acc,
    'Precision': best_precision,
    'Recall': best_recall,
    'F1 Score': best_f1,
    'AUC': best_auc,
    'MCC': best_mcc,
    'FS_len': len_best_chromo,
    'Antibiotic': antibiotic
}
df_generation = pd.DataFrame(generation_results)
df_generation.to_csv(os.path.join(ada_folder, output_dir, f"{antibiotic}_training_genresults.csv"), index=False)
print("Training results saved:")
print(df_generation)

# ============================== Save Feature Sets ==============================
for idx, chromosome in enumerate(best_chromo):
    GA_features = X.iloc[:, chromosome].copy()
    GA_features.to_csv(os.path.join(ada_folder, output_dir, "feature_set", f"{antibiotic}_fs_{idx}.csv"), index=False)
