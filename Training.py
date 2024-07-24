import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

def train_test(model, X, y, kfold):
	cv = StratifiedKFold(n_splits=kfold, shuffle=True)
	acc_local = []
	auc_local = []
	cm_local = []
	f1_local = []
	precision_local = []
	recall_local = []
	mcc_local =[]
	for train, test in cv.split(X, y):
		model.fit(X.iloc[train], y[train])
		true_labels = np.asarray(y[test])
		predictions = model.predict(X.iloc[test])
		probabilities = model.predict_proba(X.iloc[test])[:, 1]
		accuracy = accuracy_score(true_labels, predictions)
		auc = roc_auc_score(true_labels, probabilities)
		cm = confusion_matrix(true_labels, predictions)
		precision = precision_score(true_labels, predictions)
		recall = recall_score(true_labels, predictions)
		f1 = f1_score(true_labels, predictions)
		mcc = matthews_corrcoef(true_labels, predictions)
		
		auc_local.append(auc)
		acc_local.append(accuracy)
		cm_local.append(cm)
		precision_local.append(precision)
		recall_local.append(recall)
		f1_local.append(f1)
		mcc_local.append(mcc)
		
	scores = {
			'Accuracy': np.mean(acc_local),
			'AUC': np.mean(auc_local),
			'Precision': np.mean(precision_local),
			'Recall': np.mean(recall_local),
			'F1-score': np.mean(f1_local),
			'MCC': np.mean(mcc_local)
		}	
	return scores
