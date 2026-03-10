import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from sklearn.base import clone

# Define a function that trains and evaluates the model for a single fold
def train_single_fold(model, X, y, train_idx, test_idx):
	# Clone the model to avoid sharing the same instance across folds
	model_clone = clone(model)

	# Train the model
	model_clone.fit(X.iloc[train_idx], y[train_idx])

	# Predict on the test set
	true_labels = np.asarray(y[test_idx])
	predictions = model_clone.predict(X.iloc[test_idx])
	probabilities = model_clone.predict_proba(X.iloc[test_idx])[:, 1]
	
	# Calculate metrics
	accuracy = accuracy_score(true_labels, predictions)
	auc = roc_auc_score(true_labels, probabilities)
	cm = confusion_matrix(true_labels, predictions)
	precision = precision_score(true_labels, predictions)
	recall = recall_score(true_labels, predictions)
	f1 = f1_score(true_labels, predictions)
	mcc = matthews_corrcoef(true_labels, predictions)
	#print(accuracy, auc, cm, precision, recall, f1, mcc)
	return accuracy, auc, cm, precision, recall, f1, mcc
	
# Modify the original train_test function to parallelize the cross-validation
def train_test(model, X, y, kfold, n_jobs):
	cv = StratifiedKFold(n_splits=kfold, shuffle=True)
	# Use joblib to parallelize the training on each fold
	results = Parallel(n_jobs=n_jobs)(
	delayed(train_single_fold)(model, X, y, train_idx, test_idx) for train_idx, test_idx in cv.split(X, y))
	# Unpack the results from all folds
	acc_local, auc_local, cm_local, precision_local, recall_local, f1_local, mcc_local = zip(*results)
	# Calculate mean scores across all folds		
	scores = {
			'Accuracy': np.mean(acc_local),
			'AUC': np.mean(auc_local),
			'Precision': np.mean(precision_local),
			'Recall': np.mean(recall_local),
			'F1-score': np.mean(f1_local),
			'MCC': np.mean(mcc_local)
		}	
	return scores
