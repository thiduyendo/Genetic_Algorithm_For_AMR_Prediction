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
```
git clone https://github.com/thiduyendo/Genetic_Algorithm_For_AMR_Prediction.git
```
### Install Dependencies:
Ensure that you have all necessary Python libraries listed in requirement.txt installed: 
```
pip install -r requirements.txt
```
### Run the Project:
Execute main.py to start the genetic algorithm process and generate results:
```
python main.py --input_folder /path/to/input --output_folder /path/to/output
```
> **Note:**
> - input_folder: ensure to locate the trainGA_Tobramycin.csv and trainGA_ceftazidime.csv into this folder
> - output_folder: the result of the model performance
### Contributing
Contributions are welcome! Please follow the guidelines for submitting pull requests and ensure that all tests pass before submitting.
#### Fork the Repository:
Click the "Fork" button at the top right of this repository page on GitHub. This creates a copy of the repository under your own GitHub account.
#### Clone Your Fork:
- Open your terminal or command prompt.
- Clone the forked repository to your local machine:
```
git clone https://github.com/your-username/your-forked-repository.git
```
#### Navigate to the project directory:
```
cd your-forked-repository
```
#### Create a New Branch:
It’s best practice to create a new branch for each feature or bug fix
```
git checkout -b my-new-feature
```
#### Make Changes
#### Commit Your Changes:
Add the changes to the staging area:
```
git add .
git commit -m "Add a feature"
```
#### Push to the Branch:
```
git push origin feature-branch
```
#### Create a Pull Request:
- Go to the original repository on GitHub.
- Click on "New Pull Request."
- Select the branch you pushed to and compare it with the main branch of the original repository.
- Provide a clear description of your changes and submit the pull request.
### Contact:
Please don't hesitate to write me an email if you have any questions: [dtduyen1990@gmail.com](https://mail.google.com/mail/u/0/#inbox?compose=DmwnWrRttWSfKhnnWhrMtpdSmmPlQNwlvsCllVcTCMTPhdZqbBCgDGXDNtsXvwXMTdBNDZDpHCFL)
