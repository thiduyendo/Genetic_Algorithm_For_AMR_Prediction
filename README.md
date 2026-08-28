# Genetic Algorithm (GA-SVM) for Antimicrobial Resistance Prediction

## Project Overview

This project implements a **genetic algorithm (GA)-based feature selection approach combined with Support Vector Machine (SVM) classification** for predicting antimicrobial resistance (AMR) in *Pseudomonas aeruginosa*.

The approach uses evolutionary optimization to identify informative genomic features from high-dimensional pan-genome datasets. The selected features are subsequently evaluated using machine-learning models to assess their ability to predict antibiotic resistance.

The workflow was developed for genomic AMR prediction and is intended to support reproducible feature-selection and model-evaluation experiments.

---

## Key Features

* **Genetic algorithm-based feature selection**
  Uses a genetic algorithm to search for informative subsets of genomic features.

* **Machine-learning models**
  Supports machine-learning models including **Support Vector Machine (SVM)** and **XGBoost** for evaluating predictive performance.

* **High-dimensional genomic data**
  Designed for pan-genome feature matrices containing SNP and gene-cluster features.

* **Cross-validation**
  Uses stratified cross-validation to evaluate model performance during feature selection and reduce dependence on a single train/test split.

* **Multiple performance metrics**
  Model performance can be evaluated using:

  * Accuracy
  * Precision
  * Recall
  * F1 score
  * AUC (Area Under the ROC Curve)
  * MCC (Matthews Correlation Coefficient)

---

## Workflow

The overall workflow is:

```text
Pan-genome feature matrix
          │
          ▼
     Data preprocessing
          │
          ▼
 Genetic algorithm
          │
          ▼
 Feature selection
          │
          ▼
 Selected genomic features
          │
          ▼
 Machine-learning model
     (SVM / XGBoost)
          │
          ▼
   Cross-validation
          │
          ▼
 Performance evaluation
```

The genetic algorithm represents candidate feature subsets as chromosomes. Candidate solutions are evaluated according to their predictive performance, and evolutionary operations such as selection, crossover, and mutation are used to generate subsequent populations.

---

## Repository Structure

```text
Genetic_Algorithm_For_AMR_Prediction/
│
├── data/
│   └── combine/
│       ├── ADA_combine_imipenem.csv
│       └── ADA_combine_tobramycin.csv
│
├── README.md
├── requirement.txt
└── src/
    ├── GeneticAlgorithm.py
    ├── parallel_cv.py
    └── run_GA.py
```

### Data

`data/combine/` contains the genomic feature matrices used as input to the analysis.

Example:

```text
ADA_combine_{antibiotic}.csv
```

These files contain genomic features, including SNP and gene-cluster features, together with the corresponding antibiotic resistance phenotype.

For example:

```text
ADA_combine_tobramycin.csv
ADA_combine_imipenem.csv
```

Large datasets used in the original study may not be included in future public releases if they are subject to data-access restrictions.

---

## Source Code

### `src/GeneticAlgorithm.py`

Implements the genetic algorithm used for feature selection.

The genetic algorithm searches through candidate feature subsets and evaluates their predictive performance.

### `src/run_GA.py`

Main entry point for running the genetic algorithm.

This script:

1. Loads the genomic feature matrix.
2. Processes the selected antibiotic dataset.
3. Initializes the genetic algorithm.
4. Runs the specified number of generations.
5. Saves the best feature set from each generation.
6. Saves the training performance for each generation.

### `src/parallel_cv.py`

Contains the model-training and cross-validation functionality used to evaluate candidate feature subsets.

The `train_test` function performs model training and performance evaluation using cross-validation.

---

# Installation

## Requirements

The pipeline requires Python 3 and the Python packages listed in:

```text
requirement.txt
```

Install the dependencies with:

```bash
pip install -r requirement.txt
```

An `environment.yml` file may also be provided for reproducible Conda-based installation.

---

# Usage

## Run the Genetic Algorithm

From the project root:

```bash
cd Genetic_Algorithm_For_AMR_Prediction
```

Then run:

```bash
python src/run_GA.py -antibiotic tobramycin -gen_number 50 -outdir GA_results
```

### Parameters

| Parameter     | Description                             |
| ------------- | --------------------------------------- |
| `-antibiotic` | Antibiotic phenotype to analyze         |
| `-gen_number` | Number of genetic-algorithm generations |
| `-outdir`     | Directory in which GA results are saved |

For example:

```bash
python src/run_GA.py \
    -antibiotic tobramycin \
    -gen_number 50 \
    -outdir GA_results
```

Make sure that the corresponding input dataset is available under the expected `data/combine/` directory.

---

# Output

The genetic algorithm produces a result directory containing the selected feature sets and generation-level performance results.

For example:

```text
GA_results/
├── feature_set/
│   ├── tobramycin_fs_0.csv
│   ├── tobramycin_fs_1.csv
│   ├── tobramycin_fs_2.csv
│   └── ...
│
└── tobramycin_training_genresults.csv
```

### Feature sets

Files such as:

```text
tobramycin_fs_0.csv
tobramycin_fs_1.csv
...
```

contain the best chromosome/feature set identified for each generation.

The feature set from the **final generation** represents the final feature subset selected by the genetic algorithm.

### Generation-level results

The file:

```text
tobramycin_training_genresults.csv
```

contains the training/evaluation results obtained for the genetic algorithm across generations.

These results can be used to examine how model performance changes as the genetic algorithm evolves the feature subsets.

---

# Input Data Format

The input CSV should contain genomic features and the corresponding antibiotic resistance phenotype.

A typical structure is:

```text
sample_id,feature_1,feature_2,...,feature_n,phenotype
sample_1,2,0,...,5,1
sample_2,3,0,...,4,0
...
```

The exact column names and phenotype encoding should match the requirements of the corresponding analysis scripts.

Before running the pipeline on a new dataset, verify that:

* samples are represented consistently;
* genomic features are encoded numerically;
* the antibiotic phenotype is correctly defined;
* missing values are handled appropriately;
* the feature matrix contains the columns expected by the analysis scripts.

---

# Reproducibility

To reproduce an analysis:

1. Clone the repository.
2. Install the required Python dependencies.
3. Obtain the required genomic feature matrix.
4. Place the input dataset in the appropriate `data/combine/` directory.
5. Run `run_GA.py` with the desired antibiotic and number of generations.
6. Record the software version, dataset, random seed, and analysis parameters used.

Example:

```bash
git clone https://github.com/thiduyendo/Genetic_Algorithm_For_AMR_Prediction.git

cd Genetic_Algorithm_For_AMR_Prediction

pip install -r requirement.txt

python src/run_GA.py \
    -antibiotic tobramycin \
    -gen_number 50 \
    -outdir GA_results
```

Exact results may depend on the input dataset, randomization, software versions, and model parameters.

For reproduction of results reported in the associated manuscript, use the same dataset, preprocessing procedure, parameter settings, and validation strategy described in the manuscript.

---

# Citation

If you use this software, please cite the associated publication and the archived software release.

The software will be archived through **Zenodo** to provide a persistent DOI for the specific release used in the associated research.

**Zenodo DOI:** *to be added after release*

The repository also includes a `CITATION.cff` file containing machine-readable citation metadata.

---

# Contributing

Contributions and suggestions are welcome.

To contribute:

1. Fork the repository.
2. Clone your fork.
3. Create a new branch.
4. Make your changes.
5. Commit and push the changes.
6. Open a pull request.

Example:

```bash
git clone https://github.com/your-username/Genetic_Algorithm_For_AMR_Prediction.git

cd Genetic_Algorithm_For_AMR_Prediction

git checkout -b my-new-feature

git add .

git commit -m "Add new feature"

git push origin my-new-feature
```

Then open a pull request from your branch to the main repository.

---

# Contact

For questions or issues related to the software, please open an issue in the GitHub repository:

https://github.com/thiduyendo/Genetic_Algorithm_For_AMR_Prediction

For direct correspondence:

**Thi Duyen Do**
Email: [tddo1990@gmail.com](mailto:tddo1990@gmail.com)
