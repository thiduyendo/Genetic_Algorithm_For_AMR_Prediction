import os
import pandas as pd
from sklearn.svm import SVC
from Genetic_Algorithm import generations
import argparse

def process_data(input_file, output_file):
    try:
        # Configuration
        kfold = 5 
        model = SVC(kernel='linear', probability=True)

        df = pd.read_csv(input_file)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        best_chromo, best_acc, best_precision, best_recall, best_f1, best_auc, best_mcc, len_best_chromo = generations(
        size=50, n_feat=X.shape[1], crossover_rate=0.8, mutation_rate=0.05, max_gen=3, model=model, kfold=kfold, X=X, y=y)

        generation_results = {
            'Accuracy': best_acc,
            'Precision': best_precision,
            'Recall': best_recall,
            'F1 Score': best_f1,
            'AUC': best_auc,
            'MCC': best_mcc,
            'Length': len_best_chromo
        }
        #print(generation_results)
        df_generation = pd.DataFrame(generation_results)
        df_generation.to_csv(output_file, index=False)
        print(df_generation)

    except Exception as e:
        print(f"An error occurred while processing {input_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file')
    args = parser.parse_args()
    process_data(args.input_file, args.output_file)

