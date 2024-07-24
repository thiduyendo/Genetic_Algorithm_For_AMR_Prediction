# Genetic_Algorithm_For_AMR_Prediction
## Project Overview
This project leverages genetic algorithms (GAs) to select the most important features for predicting antibiotic resistance in Pseudomonas aeruginosa. By combining advanced machine learning techniques with evolutionary algorithms, this project aims to enhance the accuracy and effectiveness of resistance predictions, potentially leading to more effective treatments and interventions.
## Key Features
- Genetic Algorithm Implementation: Implements genetic algorithms to enhance feature selection and fine-tune model parameters, thereby boosting predictive accuracy and robustness. This approach employs a diverse set of evaluation metrics to minimize bias and prevent overfitting, ensuring a more comprehensive assessment of model performance.
- Machine Learning Models: Incorporates various machine learning models, including Support Vector Machines (SVM) and XGBoost, to evaluate and predict antibiotic resistance.
- Cross-Validation: Employs Stratified K-Fold cross-validation to ensure reliable performance metrics and avoid overfitting.
- Performance Metrics: Measures model performance using accuracy, precision, recall, F1 score, AUC (Area Under the Curve), and MCC (Matthews Correlation Coefficient).
## Project Components
- Training.py: Contains the train_test function that trains models and evaluates their performance using cross-validation.
- Genetic_Algorithm.py: Implements the genetic algorithm for feature selection and model optimization.
- main.py: Runs the genetic algorithm with specified parameters, processes data, and saves results.
## Data
- Input Data: SNP (Single Nucleotide Polymorphism) pan-genome and associated antibiotic resistance labels for different antibiotics.
- Output Data: CSV files containing the results of the genetic algorithm, including performance metrics for each antibiotic.
## Getting Started
### Clone the Repository:
git clone https://github.com/thiduyendo/Genetic_Algorithm_For_AMR_Prediction.git
### Install Dependencies:
Ensure that you have all necessary Python libraries listed in requirement.txt installed: 
pip install -r requirements.txt
### Run the Project:
Execute main.py to start the genetic algorithm process and generate results:
python main.py --input_folder /path/to/input --output_folder /path/to/output
- input_folder: ensure to locate the trainGA_Tobramycin.csv and trainGA_ceftazidime.csv into this folder
- output_folder: the result of the model performance
### Contributing
Contributions are welcome! Please follow the guidelines for submitting pull requests and ensure that all tests pass before submitting.
